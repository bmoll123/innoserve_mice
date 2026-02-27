from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
from model import Discriminator, Projection, PatchMaker

import numpy as np
import torch.nn.functional as F

import logging
import os
import torch
import tqdm
import common
import metrics
import cv2
import utils
import glob
import shutil
import time
import csv
try:
    from thop import profile
except ImportError:
    profile = None

LOGGER = logging.getLogger(__name__)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class TBWrapper:
    def __init__(self, log_dir):
        self.g_iter = 0
        self.logger = SummaryWriter(log_dir=log_dir)

    def step(self):
        self.g_iter += 1


class MICE(torch.nn.Module):
    def __init__(self, device):
        super(MICE, self).__init__()
        self.device = device

    def load(
            self,
            backbone,
            layers_to_extract_from,
            device,
            input_shape,
            pretrain_embed_dimension,
            target_embed_dimension,
            patchsize=3,
            patchstride=1,
            meta_epochs=640,
            eval_epochs=1,
            dsc_layers=2,
            dsc_hidden=1024,
            dsc_margin=0.5,
            train_backbone=False,
            pre_proj=1,
            k=0.25,
            lr=0.0001,
            n_neighbors=9,
            tangent_ratio=0.2,
            limit=392,
            **kwargs,
    ):
        self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape
        self.device = device

        self.forward_modules = torch.nn.ModuleDict({})
        feature_aggregator = common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device, train_backbone
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator

        preprocessing = common.Preprocessing(feature_dimensions, pretrain_embed_dimension)
        self.forward_modules["preprocessing"] = preprocessing
        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = common.Aggregator(target_dim=target_embed_dimension)
        preadapt_aggregator.to(self.device)
        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.meta_epochs = meta_epochs
        self.lr = lr
        self.train_backbone = train_backbone
        if self.train_backbone:
            self.backbone_opt = torch.optim.AdamW(self.forward_modules["feature_aggregator"].backbone.parameters(), lr)

        self.pre_proj = pre_proj
        if self.pre_proj > 0:
            self.pre_projection = Projection(self.target_embed_dimension, self.target_embed_dimension, pre_proj)
            self.pre_projection.to(self.device)
            self.proj_opt = torch.optim.Adam(self.pre_projection.parameters(), lr, weight_decay=1e-5)

        self.dsc_lr = lr * 2
        self.eval_epochs = eval_epochs
        self.dsc_layers = dsc_layers
        self.dsc_hidden = dsc_hidden
        self.discriminator = Discriminator(self.target_embed_dimension, n_layers=dsc_layers, hidden=dsc_hidden)
        self.discriminator.to(self.device)
        self.dsc_opt = torch.optim.AdamW(self.discriminator.parameters(), lr=self.dsc_lr)
        self.dsc_margin = dsc_margin

        self.n_neighbors = n_neighbors
        self.tangent_ratio = tangent_ratio
        self.memory_bank = None
        self.k = k
        self.limit = limit

        self.patch_maker = PatchMaker(patchsize, stride=patchstride)
        self.anomaly_segmentor = common.RescaleSegmentor(device=self.device, target_size=input_shape[-2:])
        self.model_dir = ""
        self.dataset_name = ""
        self.logger = None

    def set_model_dir(self, model_dir, dataset_name):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.ckpt_dir = os.path.join(self.model_dir, dataset_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.tb_dir = os.path.join(self.ckpt_dir, "tb")
        os.makedirs(self.tb_dir, exist_ok=True)
        self.logger = TBWrapper(self.tb_dir)

    def _embed(self, images, detach=True, provide_patch_shapes=False, evaluation=False):
        """Returns feature embeddings for images."""
        if not evaluation and self.train_backbone:
            self.forward_modules["feature_aggregator"].train()
            features = self.forward_modules["feature_aggregator"](images, eval=evaluation)
        else:
            self.forward_modules["feature_aggregator"].eval()
            with torch.no_grad():
                features = self.forward_modules["feature_aggregator"](images)

        features = [features[layer] for layer in self.layers_to_extract_from]

        features = [self.patch_maker.patchify(x, return_spatial_info=True) for x in features]
        patch_shapes = [x[1] for x in features]
        patch_features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(patch_features)):
            feature = patch_features[i]
            patch_dims = patch_shapes[i]

            feature = feature.reshape(
                feature.shape[0], patch_dims[0], patch_dims[1], *feature.shape[2:]
            )
            feature = feature.permute(0, 3, 4, 5, 1, 2)
            perm_base_shape = feature.shape
            feature = feature.reshape(-1, *feature.shape[-2:])
            feature = F.interpolate(
                feature.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            feature = feature.squeeze(1)
            feature = feature.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            feature = feature.permute(0, 4, 5, 1, 2, 3)
            feature = feature.reshape(len(feature), -1, *feature.shape[-3:])
            patch_features[i] = feature

        patch_features = [x.reshape(-1, *x.shape[-3:]) for x in patch_features]
        patch_features = self.forward_modules["preprocessing"](patch_features)
        patch_features = self.forward_modules["preadapt_aggregator"](patch_features)

        return patch_features, patch_shapes

    def trainer(self, train_data, test_data, name):
        state_dict = {}
        ckpt_path = glob.glob(self.ckpt_dir + '/ckpt_best*')
        if len(ckpt_path) != 0:
            LOGGER.info("Start testing, ckpt file found!")
            return 0., 0., 0., 0., 0., -1.

        def update_state_dict():
            state_dict["discriminator"] = OrderedDict({
                k: v.detach().cpu()
                for k, v in self.discriminator.state_dict().items()})
            if self.pre_proj > 0:
                state_dict["pre_projection"] = OrderedDict({
                    k: v.detach().cpu()
                    for k, v in self.pre_projection.state_dict().items()})

        # 計算 Params 與 FLOPs
        dummy_input = torch.randn(1, 3, 288, 288).to(self.device)
        
        # 計算參數量 (Parameters) - 以百萬 (M) 為單位
        total_params = sum(p.numel() for p in self.discriminator.parameters())
        if self.pre_proj > 0:
            total_params += sum(p.numel() for p in self.pre_projection.parameters())
        params_M = total_params / 1e6
        params_str = f"{params_M:.3f}M"

        # 計算 FLOPs - 以十億 (G) 為單位
        flops_G = 0.0
        flops_str = "N/A"
        if profile is not None:
            try:
                macs, _ = profile(self.forward_modules["feature_aggregator"].backbone, inputs=(dummy_input, ), verbose=False)
                flops_G = macs / 1e9
                flops_str = f"{flops_G:.3f}G"
            except Exception as e:
                LOGGER.warning(f"Failed to calculate FLOPs: {e}")

        analyze_dir = os.path.join("results", "analyze results")
        os.makedirs(analyze_dir, exist_ok=True)
        
        csv_filename = f"training_log_{name}.csv"
        csv_filepath = os.path.join(analyze_dir, csv_filename)
        
        with open(csv_filepath, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "epoch_time_s", "avg_perturbation_time_ms", "image_auroc", "pixel_auroc"])

        LOGGER.info("Initializing Memory Bank for Manifold Interpolation...")
        bank_features = []

        patches_per_image = 300 
        with torch.no_grad():
            for i, data in enumerate(tqdm.tqdm(train_data, desc="Building Memory Bank")):
                img = data["image"]
                img = img.to(torch.float).to(self.device)
                
                # 取得特徵 Embedding
                if self.pre_proj > 0:
                    outputs = self.pre_projection(self._embed(img, evaluation=False)[0])
                else:
                    outputs = self._embed(img, evaluation=False)[0]
                
                # Reshape 成 (N_patches, Feature_Dim)
                outputs = outputs.reshape(img.shape[0], -1, outputs.shape[-1]) 
                outputs = outputs.reshape(-1, outputs.shape[-1])

                # 特徵抽樣 (Subsampling)
                if outputs.shape[0] > patches_per_image:
                    indices = torch.randperm(outputs.shape[0])[:patches_per_image]
                    outputs = outputs[indices]

                bank_features.append(outputs.cpu())
            
            # 合併特徵庫
            self.memory_bank = torch.cat(bank_features, dim=0)
            
            bank_path = os.path.join(self.ckpt_dir, "memory_bank.pth")
            torch.save(self.memory_bank, bank_path)
            LOGGER.info(f"Memory Bank initialized with shape: {self.memory_bank.shape}")

            self.memory_bank = self.memory_bank.cpu()

        pbar = tqdm.tqdm(range(self.meta_epochs), unit='epoch')
        pbar_str1 = ""
        best_record = None
        
        total_training_start = time.time()
        epoch_times = []
        perturb_times_all_epochs = []

        for i_epoch in pbar:
            epoch_start = time.time()
            pbar_str, pt, pg, avg_perturb_ms = self._train_discriminator(train_data, i_epoch, pbar, pbar_str1)
            update_state_dict()
            
            epoch_end = time.time()
            epoch_duration = epoch_end - epoch_start
            epoch_times.append(epoch_duration)
            perturb_times_all_epochs.append(avg_perturb_ms)

            current_i_auroc = 0.0
            current_p_auroc = 0.0

            if (i_epoch + 1) % self.eval_epochs == 0:
                images, scores, segmentations, labels_gt, masks_gt = self.predict(test_data)
                image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro = self._evaluate(images, scores, segmentations,
                                                                                        labels_gt, masks_gt, name)
                
                current_i_auroc = image_auroc
                current_p_auroc = pixel_auroc

                self.logger.logger.add_scalar("i-auroc", image_auroc, i_epoch)
                self.logger.logger.add_scalar("p-auroc", pixel_auroc, i_epoch)

                eval_path = './results/eval/' + name + '/'
                train_path = './results/training/' + name + '/'
                if best_record is None or image_auroc + pixel_auroc > best_record[0] + best_record[2]:
                    if best_record is not None:
                        os.remove(ckpt_path_best)
                    best_record = [image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro, i_epoch]
                    ckpt_path_best = os.path.join(self.ckpt_dir, "ckpt_best_{}.pth".format(i_epoch))
                    torch.save(state_dict, ckpt_path_best)
                    shutil.rmtree(eval_path, ignore_errors=True)
                    shutil.copytree(train_path, eval_path)

                pbar_str1 = f" IAUC:{round(image_auroc * 100, 2)}({round(best_record[0] * 100, 2)})" \
                            f" PAUC:{round(pixel_auroc * 100, 2)}({round(best_record[2] * 100, 2)})" \
                            f" E:{i_epoch}({best_record[-1]})"
                pbar_str += pbar_str1
                pbar.set_description_str(pbar_str)
            
            with open(csv_filepath, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([i_epoch, f"{epoch_duration:.4f}", f"{avg_perturb_ms:.4f}", f"{current_i_auroc:.4f}", f"{current_p_auroc:.4f}"])

        total_training_end = time.time()
        total_training_duration = total_training_end - total_training_start
        avg_epoch_time = np.mean(epoch_times) if epoch_times else 0.0
        avg_perturb_time_overall = np.mean(perturb_times_all_epochs) if perturb_times_all_epochs else 0.0
        
        best_img_auroc = best_record[0] if best_record else 0.0
        best_pix_auroc = best_record[2] if best_record else 0.0
        best_epoch = best_record[-1] if best_record else -1

        print("\n" + "="*40)
        print(f" PAPER METRICS SUMMARY: {name}")
        print("="*40)
        print(f" 1. Model Params : {params_str} (FLOPs: {flops_str})")
        print(f" 2. Total Training Time : {total_training_duration:.2f} s")
        print(f" 3. Average Time per Epoch : {avg_epoch_time:.4f} s")
        print(f" 4. Perturbation Latency : {avg_perturb_time_overall:.4f} ms/batch (Micro-benchmark)")
        if best_record:
            print(f" 5. Best Image AUROC : {best_img_auroc * 100:.2f} % (at Epoch {best_epoch})")
            print(f" 6. Best Pixel AUROC : {best_pix_auroc * 100:.2f} % (at Epoch {best_epoch})")
        print("="*40 + "\n")

        final_txt_path = os.path.join(analyze_dir, "final_best_results.txt")
        result_line = (f"[{name}] Best_I-AUROC: {best_img_auroc:.4f}, Best_P-AUROC: {best_pix_auroc:.4f}, "
                    f"Time/Epoch: {avg_epoch_time:.4f}s, Latency: {avg_perturb_time_overall:.4f}ms, "
                    f"Params: {params_str}, FLOPs: {flops_str}, TotalTime: {total_training_duration:.2f}s\n")
        
        with open(final_txt_path, "a") as f:
            f.write(result_line)

        return best_record

    def _train_discriminator(self, train_data, cur_epoch, pbar, pbar_str1):
        self.forward_modules.eval()
        if self.pre_proj > 0:
            self.pre_projection.train()
        self.discriminator.train()

        all_loss, all_s_loss, all_b_loss = [], [], []
        all_p_true, all_p_fake, all_r_t, all_r_f, all_r_g = [], [], [], [], []
        
        batch_perturb_times = [] 
        
        sample_num = 0
        
        bank_subset_size = 4096 
        
        for i_iter, data_item in enumerate(train_data):
            self.dsc_opt.zero_grad()
            if self.pre_proj > 0:
                self.proj_opt.zero_grad()

            img = data_item["image"]
            img = img.to(torch.float).to(self.device)
            if self.pre_proj > 0:
                true_feats = self.pre_projection(self._embed(img, evaluation=False)[0])
            else:
                true_feats = self._embed(img, evaluation=False)[0]

            if self.memory_bank.shape[0] > bank_subset_size:
                indices = torch.randperm(self.memory_bank.shape[0])[:bank_subset_size]
                ref_bank = self.memory_bank[indices].to(self.device)
            else:
                ref_bank = self.memory_bank.to(self.device)
            
            # 開始計時 Perturbation Latency
            t0_perturb = time.time()
            
            dist_matrix = torch.cdist(true_feats, ref_bank)
            k = min(self.n_neighbors, ref_bank.shape[0])
            _, topk_indices = torch.topk(dist_matrix, k=k, dim=1, largest=False)
            
            rand_k_idx = torch.randint(0, k, (true_feats.shape[0],), device=self.device)
            selected_neighbor_indices = topk_indices[torch.arange(true_feats.shape[0]), rand_k_idx]
            selected_neighbors = ref_bank[selected_neighbor_indices]

            # 1. 計算當前特徵與選中鄰居之間的歐式距離
            dist_pairs = torch.norm(true_feats - selected_neighbors, dim=1, keepdim=True)
            
            # 2. 設定動態閾值 (例如: 取當前 Batch 距離分佈的 90 百分位數)
            mixup_threshold = torch.quantile(dist_pairs, 0.9)

            # 3. 初始隨機 Lambda [0, 0.5]
            lam = torch.rand(true_feats.shape[0], 1, device=self.device) * 0.5
            
            # 4. 應用限制: 若距離 > 閾值，則強制 lam = 0 (不進行插值)
            mask_too_far = dist_pairs > mixup_threshold
            lam[mask_too_far] = 0.0

            local_centers = lam * true_feats + (1 - lam) * selected_neighbors
            
            # A. 取得徑向方向 (Radial Direction)
            direct = true_feats - local_centers
            d_ct = torch.norm(direct, dim=1, keepdim=True) + 1e-8
            unit_direct = direct / d_ct
            r_ct = d_ct.mean()
            
            # B. 生成切向擾動 (Tangential Perturbation)
            if hasattr(self, 'tangent_ratio') and self.tangent_ratio > 0:
                # 生成隨機高斯噪聲向量
                rand_vec = torch.randn_like(direct)
                
                # Gram-Schmidt 正交化：投影並扣除，只保留垂直於徑向的分量
                proj = (rand_vec * unit_direct).sum(dim=1, keepdim=True) * unit_direct
                tangential = rand_vec - proj
                
                # 歸一化切向向量
                tangential = tangential / (torch.norm(tangential, dim=1, keepdim=True) + 1e-8)
                
                # 合成位移向量 (Displacement) = 單位徑向 + 比例 * 單位切向
                # 這會形成一個朝外的橢圓錐方向
                displacement = unit_direct + self.tangent_ratio * tangential
            else:
                # 如果 ratio 為 0，退化回原本的純徑向延伸
                displacement = unit_direct

            # C. 生成異常特徵
            # 保持原本的 logic: 從 true_feats 出發，加上 (方向 * 距離)
            # 距離 = r_ct * self.k
            perturbation = (displacement * r_ct * self.k).detach()
            fake_feats = true_feats + perturbation

            t1_perturb = time.time()
            batch_perturb_times.append((t1_perturb - t0_perturb) * 1000) # 轉換為 ms

            d_tf = torch.norm(fake_feats - true_feats, dim=1, keepdim=True)
            r_tf = d_tf.mean()
            d_cf = torch.norm(fake_feats - local_centers, dim=1, keepdim=True)
            r_cf = d_cf.mean()
            svdd_loss = r_ct 

            scores = self.discriminator(torch.concat([true_feats, fake_feats]))
            true_scores = scores[:len(true_feats)]
            fake_scores = scores[len(true_feats):]
            true_loss = torch.nn.BCELoss()(true_scores, torch.zeros_like(true_scores))
            fake_loss = torch.nn.BCELoss()(fake_scores, torch.ones_like(fake_scores))
            bce_loss = true_loss + fake_loss

            loss = svdd_loss + bce_loss
            loss.backward()
            if self.pre_proj > 0:
                self.proj_opt.step()
            if self.train_backbone:
                self.backbone_opt.step()
            self.dsc_opt.step()

            pix_true = true_scores.detach()
            pix_fake = fake_scores.detach()
            p_t = (pix_true < self.dsc_margin).sum() / pix_true.shape[0]
            p_g = (pix_fake >= self.dsc_margin).sum() / pix_fake.shape[0]

            self.logger.logger.add_scalar("total_loss", loss, self.logger.g_iter)
            self.logger.step()

            all_loss.append(loss.detach().cpu().item())
            all_s_loss.append(svdd_loss.detach().cpu().item())
            all_b_loss.append(bce_loss.detach().cpu().item())
            all_p_true.append(p_t.cpu().item())
            all_p_fake.append(p_g.cpu().item())
            all_r_t.append(r_ct.cpu().item())
            all_r_g.append(r_cf.cpu().item())
            all_r_f.append(r_tf.cpu().item())

            sample_num += img.shape[0]

            all_s_loss_ = np.mean(all_s_loss)
            all_b_loss_ = np.mean(all_b_loss)
            all_p_true_ = np.mean(all_p_true)
            all_p_fake_ = np.mean(all_p_fake)
            all_r_t_ = np.mean(all_r_t)
            all_r_g_ = np.mean(all_r_g)
            all_r_f_ = np.mean(all_r_f)

            pbar_str = f"epoch:{cur_epoch}"
            pbar_str += f" sl:{all_s_loss_:.2e}"
            pbar_str += f" bl:{all_b_loss_:.2e}"
            pbar_str += f" pt:{all_p_true_ * 100:.2f}"
            pbar_str += f" pf:{all_p_fake_ * 100:.2f}"
            pbar_str += f" c->t:{all_r_t_:.2f}"
            pbar_str += f" c->f:{all_r_g_:.2f}"
            pbar_str += f" t->f:{all_r_f_:.2f}"
            pbar_str += f" sample:{sample_num}"
            pbar_str2 = pbar_str
            pbar_str += pbar_str1
            pbar.set_description_str(pbar_str)

            if sample_num > self.limit:
                break
        avg_perturb_time_ms = np.mean(batch_perturb_times) if batch_perturb_times else 0.0
        return pbar_str2, all_p_true_, all_p_fake_, avg_perturb_time_ms

    def tester(self, test_data, name):
        ckpt_path = glob.glob(self.ckpt_dir + '/ckpt_best*')
        if len(ckpt_path) != 0:
            state_dict = torch.load(ckpt_path[0], map_location=self.device)
            if 'discriminator' in state_dict:
                self.discriminator.load_state_dict(state_dict['discriminator'])
                if "pre_projection" in state_dict:
                    self.pre_projection.load_state_dict(state_dict["pre_projection"])
            else:
                self.load_state_dict(state_dict, strict=False)

            images, scores, segmentations, labels_gt, masks_gt = self.predict(test_data)
            image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro = self._evaluate(images, scores, segmentations,
                                                                                     labels_gt, masks_gt, name, path='eval')
            epoch = int(ckpt_path[0].split('_')[-1].split('.')[0])
        else:
            LOGGER.info("No ckpt file found!")
            return 0., 0., 0., 0., 0., -1.

        return image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro, epoch

    def _evaluate(self, images, scores, segmentations, labels_gt, masks_gt, name, path='training'):
        scores = np.squeeze(np.array(scores))
        image_scores = metrics.compute_imagewise_retrieval_metrics(scores, labels_gt, path)
        image_auroc = image_scores["auroc"]
        image_ap = image_scores["ap"]

        segmentations = np.array(segmentations)
        pixel_scores = metrics.compute_pixelwise_retrieval_metrics(segmentations, masks_gt, path)
        pixel_auroc = pixel_scores["auroc"]
        pixel_ap = pixel_scores["ap"]
        if path == 'eval':
            try:
                pixel_pro = metrics.compute_pro(np.squeeze(np.array(masks_gt)), segmentations)
            except:
                pixel_pro = 0.
        else:
            pixel_pro = 0.

        defects = images 
        targets = masks_gt

        save_limit = min(len(defects), 50) 

        ng_indices = [i for i, target in enumerate(targets) if target.sum() > 0]
        ok_indices = [i for i, target in enumerate(targets) if target.sum() == 0]

        half_limit = save_limit // 2

        ok_take = min(len(ok_indices), half_limit)
        ng_take = min(len(ng_indices), half_limit)

        if ok_take < half_limit:
            ng_take = min(len(ng_indices), save_limit - ok_take)
        elif ng_take < half_limit:
            ok_take = min(len(ok_indices), save_limit - ng_take)

        save_indices = ok_indices[:ok_take] + ng_indices[:ng_take]
        
        for idx, orig_idx in enumerate(save_indices):
            defect = defects[orig_idx]
            
            target_mask = targets[orig_idx].astype(np.uint8)
            if target_mask.shape[0] == 1:
                target_mask = target_mask.transpose([1, 2, 0])
                target_mask = np.repeat(target_mask, 3, axis=-1)
            target = target_mask * 255

            mask = cv2.cvtColor(cv2.resize(segmentations[orig_idx].astype(np.float32), (defect.shape[1], defect.shape[0])),
                                cv2.COLOR_GRAY2BGR)
            mask = (mask * 255).astype('uint8')
            mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

            img_up = np.hstack([defect, target, mask])
            img_up = cv2.resize(img_up, (256 * 3, 256))
            full_path = './results/' + path + '/' + name + '/'
            utils.del_remake_dir(full_path, del_flag=False)

            label_str = "NG" if orig_idx in ng_indices else "OK"
            cv2.imwrite(full_path + str(idx + 1).zfill(3) + f'_{label_str}_img{orig_idx}.png', img_up)

        return image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro

    def predict(self, test_dataloader):
        """This function provides anomaly scores/maps for full dataloaders."""
        self.forward_modules.eval()

        img_paths = []
        images = []
        scores = []
        masks = []
        labels_gt = []
        masks_gt = []

        with tqdm.tqdm(test_dataloader, desc="Inferring...", leave=False, unit='batch') as data_iterator:
            for data in data_iterator:
                if isinstance(data, dict):
                    labels_gt.extend(data["is_anomaly"].cpu().numpy())
                    if data.get("mask_gt", None) is not None:
                        masks_gt.append(data["mask_gt"].cpu().numpy().astype(np.bool_)) 
                    image = data["image"]
                    
                    for img_arr in image.cpu().numpy():
                        images.append(utils.torch_format_2_numpy_img(img_arr))
                        
                    img_paths.extend(data["image_path"])
                _scores, _masks = self._predict(image)
                scores.extend(_scores)
                
                # 將預測的 mask 轉為 float16
                masks.extend([m.astype(np.float16) for m in _masks])

        if len(masks_gt) > 0:
            masks_gt = np.concatenate(masks_gt, axis=0)

        return images, scores, masks, labels_gt, masks_gt

    def _predict(self, img):
        """Infer score and mask for a batch of images."""
        img = img.to(torch.float).to(self.device)
        self.forward_modules.eval()

        if self.pre_proj > 0:
            self.pre_projection.eval()
        self.discriminator.eval()

        with torch.no_grad():
            patch_features, patch_shapes = self._embed(img, provide_patch_shapes=True, evaluation=True)
            if self.pre_proj > 0:
                patch_features = self.pre_projection(patch_features)
                patch_scores = image_scores = self.discriminator(patch_features)

            patch_scores = self.patch_maker.unpatch_scores(patch_scores, batchsize=img.shape[0])
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(img.shape[0], scales[0], scales[1])
            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)

            image_scores = self.patch_maker.unpatch_scores(image_scores, batchsize=img.shape[0])
            image_scores = self.patch_maker.score(image_scores)
            if isinstance(image_scores, torch.Tensor):
                image_scores = image_scores.cpu().numpy()

        return list(image_scores), list(masks)
