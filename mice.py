from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from model import Discriminator, Projection, PatchMaker
from pathlib import Path

import numpy as np
import torch.nn.functional as F
import PIL.Image

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
            top_k=1,
            thr_mode="fixed",
            thr_percentile=99.0,
            blur_sigma=4,
            accum_images=1,
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
        # dsc_margin 同時是訓練統計 (pt/pf) 的門檻與分類判定門檻。
        # thr_mode='percentile' 時，判定門檻改由訓練集 (全正常) 的分數分布決定，
        # self.dsc_margin 仍然維持原值給 pt/pf 用。
        self.thr_mode = thr_mode
        self.thr_percentile = thr_percentile
        self.threshold = dsc_margin

        self.n_neighbors = n_neighbors
        self.tangent_ratio = tangent_ratio
        self.memory_bank = None
        self.k = k
        self.limit = limit
        # 累積這麼多張圖的梯度才更新一次權重 (gradient accumulation)，
        # 等效於把 batch size 放大到 accum_images，但不用真的一次塞更多圖進 GPU。
        # 1 = 跟原本一樣，每個 batch 都更新。
        self.accum_images = max(1, accum_images)

        self.patch_maker = PatchMaker(patchsize, top_k=top_k, stride=patchstride)
        self.anomaly_segmentor = common.RescaleSegmentor(device=self.device, target_size=input_shape[-2:],
                                                          smoothing=blur_sigma)
        self.model_dir = ""
        self.dataset_name = ""
        self.logger = None

    def set_model_dir(self, model_dir, dataset_name, results_path="results"):
        # results_path 是本次實驗的根目錄 (由 --results_path 指定)。
        # eval / training / analyze results 全部掛在它底下，不同實驗才不會互相覆蓋。
        self.results_path = results_path
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

        analyze_dir = os.path.join(self.results_path, "analyze results")
        os.makedirs(analyze_dir, exist_ok=True)
        
        csv_filename = f"training_log_{name}.csv"
        csv_filepath = os.path.join(analyze_dir, csv_filename)
        
        with open(csv_filepath, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "epoch_time_s", "avg_perturbation_time_ms", "image_auroc", "pixel_auroc",
                             "acc", "balanced_acc", "f1", "recall_fake", "recall_real",
                             "best_acc", "best_balanced_acc", "best_threshold",
                             "best_f1", "best_f1_threshold"])

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
            current_cls = {}

            if (i_epoch + 1) % self.eval_epochs == 0:
                if self.thr_mode == "percentile":
                    self.threshold = self.calibrate_threshold(train_data)
                images, scores, segmentations, labels_gt, masks_gt, img_paths = self.predict(test_data)
                image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro, cls = self._evaluate(
                    images, scores, segmentations, labels_gt, masks_gt, name, img_paths=img_paths)

                current_i_auroc = image_auroc
                current_p_auroc = pixel_auroc
                current_cls = cls

                self.logger.logger.add_scalar("i-auroc", image_auroc, i_epoch)
                self.logger.logger.add_scalar("p-auroc", pixel_auroc, i_epoch)
                self.logger.logger.add_scalar("acc", cls["acc"], i_epoch)
                self.logger.logger.add_scalar("balanced-acc", cls["balanced_acc"], i_epoch)

                eval_path = os.path.join(self.results_path, 'eval', name) + '/'
                train_path = os.path.join(self.results_path, 'training', name) + '/'
                if best_record is None or image_auroc + pixel_auroc > best_record[0] + best_record[2]:
                    if best_record is not None:
                        os.remove(ckpt_path_best)
                    best_record = [image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro, i_epoch]
                    ckpt_path_best = os.path.join(self.ckpt_dir, "ckpt_best_{}.pth".format(i_epoch))
                    torch.save(state_dict, ckpt_path_best)
                    shutil.rmtree(eval_path, ignore_errors=True)
                    shutil.copytree(train_path, eval_path)

                pbar_str1 = f" IAUC:{round(image_auroc * 100, 2)}({round(best_record[0] * 100, 2)})" \
                            f" ACC:{round(cls['acc'] * 100, 2)}" \
                            f" BACC:{round(cls['balanced_acc'] * 100, 2)}" \
                            f" E:{i_epoch}({best_record[-1]})"
                pbar_str += pbar_str1
                pbar.set_description_str(pbar_str)
            
            with open(csv_filepath, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([i_epoch, f"{epoch_duration:.4f}", f"{avg_perturb_ms:.4f}",
                                 f"{current_i_auroc:.4f}", f"{current_p_auroc:.4f}"] +
                                [f"{current_cls.get(key, 0.):.4f}" for key in
                                 ("acc", "balanced_acc", "f1", "recall_fake", "recall_real",
                                  "best_acc", "best_balanced_acc", "best_threshold",
                                  "best_f1", "best_f1_threshold")])

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

        # ── Gradient accumulation ────────────────────────────────
        # accum_steps 用第一個 batch 的實際大小換算，累積這麼多個 batch 的梯度
        # 才呼叫一次 optimizer.step()，等效於把 batch size 放大成 accum_images，
        # 不用真的一次塞更多張圖進 GPU。每個 batch 的 loss 除以 accum_steps 再
        # backward()，梯度才會是「平均」而不是「加總放大 accum_steps 倍」。
        accum_steps = 1
        accum_count = 0
        self.dsc_opt.zero_grad()
        if self.pre_proj > 0:
            self.proj_opt.zero_grad()

        for i_iter, data_item in enumerate(train_data):
            img = data_item["image"]
            img = img.to(torch.float).to(self.device)
            if i_iter == 0:
                accum_steps = max(1, round(self.accum_images / img.shape[0]))
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
            (loss / accum_steps).backward()
            accum_count += 1

            # 累積夠 accum_steps 個 batch (等效 accum_images 張圖) 才真正更新權重，
            # 或這是這個 epoch 最後一個 batch 時強制更新一次 (不然剩下的梯度會被丟掉)。
            is_last_batch = (i_iter == len(train_data) - 1)
            if accum_count >= accum_steps or is_last_batch:
                if self.pre_proj > 0:
                    self.proj_opt.step()
                if self.train_backbone:
                    self.backbone_opt.step()
                self.dsc_opt.step()

                self.dsc_opt.zero_grad()
                if self.pre_proj > 0:
                    self.proj_opt.zero_grad()
                accum_count = 0

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
                if accum_count > 0:
                    # --limit 提早結束這個 epoch，把還沒套用的累積梯度補做最後一次更新，
                    # 不然這幾個 batch 的梯度會被靜靜丟掉。
                    if self.pre_proj > 0:
                        self.proj_opt.step()
                    if self.train_backbone:
                        self.backbone_opt.step()
                    self.dsc_opt.step()
                    self.dsc_opt.zero_grad()
                    if self.pre_proj > 0:
                        self.proj_opt.zero_grad()
                    accum_count = 0
                break
        avg_perturb_time_ms = np.mean(batch_perturb_times) if batch_perturb_times else 0.0
        return pbar_str2, all_p_true_, all_p_fake_, avg_perturb_time_ms

    def calibrate_threshold(self, train_data, limit=200):
        """
        用訓練集 (全部是正常樣本) 的分數分布決定判定門檻。

        取第 thr_percentile 百分位 —— 意思是「容許 1% 的正常樣本被誤報」。
        完全不碰 test 標籤，所以這是可部署的門檻，不像 best_threshold 那樣
        是用測試答案挑出來的樂觀上界。

        會這樣做是因為 discriminator 的輸出分數會整體漂移: DeepPCB 上
        正常與異常的分數都擠在 0.9 以上，固定門檻 0.5 就切不到任何東西。
        """
        self.forward_modules.eval()
        scores, seen = [], 0
        with torch.no_grad():
            for data in train_data:
                img = data["image"] if isinstance(data, dict) else data
                s, _ = self._predict(img)
                scores.extend(np.asarray(s).ravel().tolist())
                seen += img.shape[0]
                if seen >= limit:
                    break
        if not scores:
            return self.threshold
        thr = float(np.percentile(scores, self.thr_percentile))
        LOGGER.info(f"Calibrated threshold = {thr:.4f} "
                    f"(p{self.thr_percentile} of {len(scores)} normal train scores)")
        return thr

    def tester(self, test_data, name, train_data=None):
        ckpt_path = glob.glob(self.ckpt_dir + '/ckpt_best*')
        if len(ckpt_path) != 0:
            state_dict = torch.load(ckpt_path[0], map_location=self.device)
            if 'discriminator' in state_dict:
                self.discriminator.load_state_dict(state_dict['discriminator'])
                if "pre_projection" in state_dict:
                    self.pre_projection.load_state_dict(state_dict["pre_projection"])
            else:
                self.load_state_dict(state_dict, strict=False)

            if self.thr_mode == "percentile" and train_data is not None:
                self.threshold = self.calibrate_threshold(train_data)

            images, scores, segmentations, labels_gt, masks_gt, img_paths = self.predict(test_data)
            image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro, cls = self._evaluate(
                images, scores, segmentations, labels_gt, masks_gt, name, path='eval', img_paths=img_paths)
            epoch = int(ckpt_path[0].split('_')[-1].split('.')[0])

            print("\n" + "=" * 46)
            print(f" CLASSIFICATION REPORT: {name}   (n={cls['n']})")
            print("=" * 46)
            if self.single_class_test:
                print(" test set 只有一種標籤 -> 無法計算 AUROC / accuracy。")
                print(" 請看 analyze results/predictions_*.csv 的 score 排序與 heatmap。")
                print("=" * 46 + "\n")
                return image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro, epoch, 0., 0.
            print(f" threshold ({self.thr_mode:^10}) : {cls['threshold']:.4f}")
            print(f" Accuracy               : {cls['acc'] * 100:.2f} %")
            print(f" Balanced Accuracy      : {cls['balanced_acc'] * 100:.2f} %")
            print(f" F1 (fake)              : {cls['f1'] * 100:.2f} %")
            print(f" Recall  fake / real    : {cls['recall_fake'] * 100:.2f} % / {cls['recall_real'] * 100:.2f} %")
            print(f" Confusion  TP/FP/FN/TN : {cls['tp']}/{cls['fp']}/{cls['fn']}/{cls['tn']}")
            print(f" -- oracle threshold {cls['best_threshold']:.3f}: "
                  f"acc {cls['best_acc'] * 100:.2f} % / bacc {cls['best_balanced_acc'] * 100:.2f} % (樂觀上界)")
            print(f" -- oracle threshold {cls['best_f1_threshold']:.3f}: "
                  f"best F1 {cls['best_f1'] * 100:.2f} % (樂觀上界)")
            print("=" * 46 + "\n")
        else:
            LOGGER.info("No ckpt file found!")
            return 0., 0., 0., 0., 0., -1., 0., 0.

        return image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro, epoch, cls["best_f1"], cls["best_f1_threshold"]

    def final_test(self, group_dir, group_id, resize, imagesize, save_visualizations=True,
                   min_box_area=200, pix_thr_mode="f1", exclude_val_good_pairs=False,
                   dataset_layout="mice"):
        """
        最終測試:對 test/good + test/defect + other_fake「全部」圖跑一次推論，
        產生這個 group 的正式報告(取代舊的 visualize_all)。

        跟 tester() 內部那次驗證(_evaluate, path='eval')不一樣:
          - 驗證集是 1:1 平衡的 test/good vs test/defect，訓練過程中每個 epoch
            拿來校準門檻/挑 best ckpt。
          - 這裡是「展開」成 test/good vs (test/defect + other_fake) 的完整測試，
            訓練完只跑一次。
          - 門檻: thr_mode='fixed'/'percentile' 時沿用 self.threshold (跟驗證集
            用同一顆，才是誠實的可部署數字，沒有用這批 test 的答案去挑)；
            thr_mode='oracle_f1'/'oracle_acc' 時**直接在這批展開後的 test 分數上
            重新搜尋**(不是沿用驗證集搜出來的值)，因為 oracle 本來就是要回答
            「這批 test 資料最好能到多少」，用驗證集那批小樣本搜出來的門檻套到
            這裡不見得是這批 test 上真正最佳的切點，兩邊都是樂觀上界，但拿哪批
            資料的答案去挑，上界會不一樣。

        輸出大部分收在 results_path/analyze results/<group_id>/ (不加 val_ 前綴，
        跟 tester() 驗證集那份 val_report.txt/val_predictions.csv 分開放):
          report.txt / predictions.csv / confusion_matrix.png
          wrong/          判斷錯誤的圖，六聯圖，檔名 {OK_or_NG}_{來源}{id}.png
          visualize_all/  全部圖，六聯圖，檔名同上

        bbox_top10 例外，統一收在 results_path/bbox_top10/<group_id>/
        (不是散在每個 group 自己的 analyze results/<group_id>/ 底下)，方便一次
        瀏覽所有 group 的框選結果:
          box 配對 F1 最高的 10 張，檔名 {排名}_{f1}_{來源}{id}.png。每個 predict
          box 跟 GT box 做 IoU>=0.3 配對 (不用非常準確，但框太大/太偏配不上、
          框的數量跟 GT 對不上都會扣分)，排名跟 mean F1 也會寫進 report.txt 最後一段

        六聯圖排版 (3 列 2 欄，輸出是 bbox 不是 segmentation 填色):
          原圖              | 原圖 + GT bbox (綠框)
          原圖 + predict bbox (紅框) | 原圖 + GT(綠) + predict(紅) 疊在一起比對
          heatmap           | heatmap 疊在原圖上

        bbox 做法: 把二值化後的 predict_mask 做連通元件 (cv2.findContours)，
        每一塊各自取外接矩形當作一個 predict bbox；GT 因為 to_mice.py 本來就是
        用矩形填出來的 (DeepPCB 只有 bbox 標註，沒有真正的瑕疵形狀)，直接對
        GT mask 做同樣的連通元件也能還原出原始 bbox。前景 IoU 用像素級
        intersection/union 算，天生就處理了一張圖有多個瑕疵框的情況，不需要
        額外做 box-to-box 配對。

        來源縮寫: g=test/good, d=test/defect, o=other_fake。同一個 stem 在
        good/defect 之間可能重複 (範本圖跟它的瑕疵版本共用檔名)，所以檔名一定要
        帶來源縮寫，不能只用 id，否則會互相覆蓋。

        exclude_val_good_pairs: 對應 main.py 的 --exclude_val_good_pairs。to_mice.py
        產生資料時，同一個 stem 的範本 (temp/good) 和瑕疵版本 (test/defect 或
        other_fake) 是同一塊 PCB 位置的成對影像，兩邊一定都各自輸出一份。而
        val(訓練過程中用來校準門檻/挑 best ckpt)讀的就是 test/good+test/defect
        這批圖，跟這裡 final_test 展開後的 test 是同一批圖的父集——用 val 選出來的
        門檻/checkpoint 再套到同一批圖上算最終指標，等於用同樣的資料選模型又拿來
        評分。開了這個 flag 後，會把 test/good 裡每個 stem 對應的 test/defect、
        other_fake 圖直接排除，不納入這次 final_test，降低這種耦合。

        dataset_layout: 'mice' (預設) 或 'yolo'。'yolo' 是給 deeppcb_yolo 那份
        資料集用的——結構是 group_dir/test/good/image、group_dir/test/defect/
        {image,label,origin_label}。label 是給 YOLO 訓練用的 txt (class cx cy
        w h 正規化 0~1)，origin_label 是 prepare_deeppcb_yolo.py 從
        deeppcb_mice 原始 mask png 直接複製過來的(不是從 txt box 還原，
        避免正規化取整造成的誤差)，這裡讀的就是 origin_label。其餘邏輯
        (門檻搜尋/AP/Miss Rate/bbox_top10/final_output)完全共用，不用另外
        寫一份。deeppcb_yolo 的 test 已經是 val 加上 other_fake 扣掉被抽去
        train/defect 的部分，這裡不用也不該再用 exclude_val_good_pairs
        (那是給 deeppcb_mice 原始結構、且 train 真的被加了 test/good 圖時
        才需要處理的洩漏)。
        """
        group_dir = Path(group_dir)
        pred_dir = Path(self.results_path) / "analyze results" / group_id
        pred_dir.mkdir(parents=True, exist_ok=True)
        viz_dir = pred_dir / "visualize_all"
        wrong_dir = pred_dir / "wrong"

        img_tf = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        mask_tf = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ])

        # (子資料夾, 對應的 GT mask 資料夾或 None=一律視為無瑕疵, 來源縮寫, image-level label)
        if dataset_layout == "yolo":
            sources = [
                ("test/good/image", None, "g", 0),
                ("test/defect/image", "test/defect/origin_label", "d", 1),
            ]
        else:
            sources = [
                ("test/good", None, "g", 0),
                ("test/defect", "ground_truth/defect", "d", 1),
                ("other_fake", "other_fake_masks", "o", 1),
            ]

        self.forward_modules.eval()
        if self.pre_proj > 0:
            self.pre_projection.eval()
        self.discriminator.eval()

        excluded_stems = set()
        if exclude_val_good_pairs:
            good_dir = group_dir / "test/good"
            if good_dir.is_dir():
                excluded_stems = {p.stem for p in good_dir.iterdir()
                                  if p.suffix.lower() in (".jpg", ".jpeg", ".png")}

        # ── 第一階段: 跑推論，把結果都留著 (score + seg map + GT) ──────
        items = []  # (src_tag, label, stem, path, orig_bgr, score, seg_map, gt_arr)
        n_excluded = 0
        predict_times = []  # 純 self._predict() 單張耗時 (秒)，不含前處理/門檻計算
        for sub, mask_sub, src_tag, label in sources:
            img_dir = group_dir / sub
            if not img_dir.is_dir():
                continue
            paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
            if excluded_stems and sub != "test/good":
                keep_paths = [p for p in paths if p.stem not in excluded_stems]
                n_excluded += len(paths) - len(keep_paths)
                paths = keep_paths

            for p in tqdm.tqdm(paths, desc=f"infer {sub}", leave=False):
                pil_img = PIL.Image.open(p).convert("RGB")
                img_t = img_tf(pil_img).unsqueeze(0)

                if img_t.device.type == "cuda" or self.device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    score, seg = self._predict(img_t)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                predict_times.append(time.perf_counter() - t0)
                score = float(np.asarray(score).ravel()[0])
                seg = seg[0]  # (H, W)，已經是 self.input_shape 的解析度

                if mask_sub is not None and (group_dir / mask_sub / f"{p.stem}.png").exists():
                    gt = PIL.Image.open(group_dir / mask_sub / f"{p.stem}.png").convert("L")
                    gt_arr = (mask_tf(gt).numpy()[0] * 255).astype(np.uint8)
                else:
                    gt_arr = np.zeros(seg.shape, dtype=np.uint8)

                orig = utils.torch_format_2_numpy_img(img_t[0].cpu().numpy())
                items.append((src_tag, label, p.stem, str(p), orig, score, seg, gt_arr))

        if exclude_val_good_pairs:
            LOGGER.info(f"final_test: exclude_val_good_pairs 已排除 {n_excluded} 張瑕疵圖 "
                        f"(跟被加進 train 的 test/good 同 stem，避免洩漏)")

        if len(predict_times) > 3:
            warm = predict_times[3:]  # 前 3 張通常有 CUDA/cudnn 暖機開銷，排除
            mean_ms = float(np.mean(warm)) * 1000
            median_ms = float(np.median(warm)) * 1000
            LOGGER.info(f"final_test: 單張 inference (純 self._predict()，不含前處理/門檻搜尋) "
                        f"mean={mean_ms:.2f}ms median={median_ms:.2f}ms "
                        f"(n={len(warm)}，已排除前 3 張暖機)")

        if not items:
            LOGGER.info(f"final_test: {group_dir} 底下找不到圖")
            return None

        scores = np.array([it[5] for it in items])
        labels_gt = np.array([it[1] for it in items])
        img_paths = [it[3] for it in items]
        segmentations = np.stack([it[6] for it in items])
        masks_gt = (np.stack([it[7] for it in items]) > 0).astype(np.uint8)

        # ── 決定門檻 ──────────────────────────────────────────────
        # fixed/percentile: 沿用 self.threshold (跟驗證集同一顆，可部署)。
        # oracle_f1/oracle_acc: 直接在這批展開後的 test 分數上重新搜，不是沿用
        # 驗證集搜出來的值 —— oracle 本來就該回答「這批 test 最好能到多少」。
        if self.thr_mode == "oracle_f1":
            test_threshold = metrics.search_best_threshold(scores, labels_gt, criterion="f1")["threshold"]
        elif self.thr_mode == "oracle_acc":
            test_threshold = metrics.search_best_threshold(scores, labels_gt, criterion="acc")["threshold"]
        else:
            test_threshold = self.threshold

        cls = metrics.compute_classification_metrics(scores, labels_gt, threshold=test_threshold)

        # ── 門檻無關指標: image/pixel AUROC、PRO，一樣算在這批展開後的 test 上 ──
        img_ret = metrics.compute_imagewise_retrieval_metrics(scores, labels_gt, path='eval')
        pix_ret = metrics.compute_pixelwise_retrieval_metrics(segmentations, masks_gt, path='eval')
        try:
            pro_masks, pro_segs = masks_gt, segmentations
            pro_limit = 200
            if len(pro_masks) > pro_limit:
                sel = np.random.default_rng(0).choice(len(pro_masks), pro_limit, replace=False)
                pro_masks, pro_segs = pro_masks[sel], pro_segs[sel]
            pixel_pro = metrics.compute_pro(pro_masks, pro_segs)
        except Exception:
            pixel_pro = 0.

        # ── segmentation -> bbox 工具: 對二值化 predict_mask 做連通元件，每一塊
        #    各自取外接矩形。GT 本身在 to_mice.py 就是用 bbox 填出來的矩形，不是
        #    真正的瑕疵形狀，對 GT mask 做同樣的連通元件也能還原出原始 bbox。
        RED, GREEN, ORANGE = (0, 0, 255), (0, 255, 0), (0, 165, 255)
        # clean=True (只給 predict_mask 用，GT 不動): open 去掉孤立小雜訊點，
        # close 把同一個瑕疵被 Gaussian blur 切碎的鄰近小區塊補起來合併，
        # 再丟掉面積太小的連通元件 —— 這三步是專門用來解決「零星小框框」問題的。
        MIN_BOX_AREA = min_box_area  # px^2，資料集是 640x640，小於這個面積視為雜訊
        MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        def mask_to_boxes(mask_bin, clean=False):
            m = mask_bin.astype(np.uint8) * 255
            if clean:
                m = cv2.morphologyEx(m, cv2.MORPH_OPEN, MORPH_KERNEL)
                m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, MORPH_KERNEL)
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boxes = [cv2.boundingRect(c) for c in contours]
            if clean:
                boxes = [b for b in boxes if b[2] * b[3] >= MIN_BOX_AREA]
            return boxes

        def draw_boxes(img, boxes, color):
            out = img.copy()
            for (x, y, w, h) in boxes:
                cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
            return out

        # GT box 不受門檻影響，先算好一次，minrisk 搜門檻跟後面的 AP/miss/
        # false alarm/bbox_top10 都直接重用，不用每次重算。
        gt_boxes_per_item = [mask_to_boxes(it[7] > 0, clean=False) for it in items]

        # ── pixel 門檻: 這個 group 專屬，用來把 predict_mask 二值化。不是
        #    test_threshold —— 那是 image-level 門檻，套在像素上幾乎必定全黑：
        #    image score = 該圖所有 patch 取 max，天生比大多數像素分數高一截。
        #
        # 曾經試過「每張圖各自搜一顆 oracle 門檻」(用那張圖自己的 GT 反推)，
        # 實測 (90100 這組) mIoU/AP 幾乎沒變 (0.2706->0.2824, 19.33%->19.36%)，
        # 但每個 group 要多跑 1~10 分鐘。結論: 框圈不準的瓶頸不是門檻選多少，
        # 是 segmentation map 本身定位不夠準 (Gaussian blur 把分數峰值抹開)。
        #
        # pix_thr_mode="f1": 群組共用一顆，pixel-level F1 準則，快。
        # pix_thr_mode="minrisk": 直接搜「讓 2*miss_rate + false_alarm 最低」
        # 的門檻——這才是真正要優化的目標，但每個候選門檻都要對全部有 GT 的圖
        # 重新切框、跑一次 match_miss_false_alarm，候選數(預設到 500)乘上圖數，
        # 明顯比 F1 版慢很多 (使用者已確認接受這個代價，換取更準的搜索)。
        if pix_thr_mode == "minrisk":
            scores_pool = segmentations.ravel().astype(np.float32)
            max_pixels = 2_000_000
            if scores_pool.size > max_pixels:
                rng = np.random.default_rng(0)
                scores_pool = scores_pool[rng.integers(0, scores_pool.size, max_pixels)]
            candidates = np.unique(scores_pool)
            if len(candidates) > 500:
                candidates = np.quantile(candidates, np.linspace(0, 1, 500))

            gt_items = [(idx, gb) for idx, gb in enumerate(gt_boxes_per_item) if gb]
            best_score, best_thr = None, float(candidates[0]) if len(candidates) else 0.5
            for t in tqdm.tqdm(candidates, desc="pixel thr (minrisk)", leave=False):
                total_tp = total_fp = total_fn = 0
                for idx, gt_boxes_i in gt_items:
                    pred_boxes_i = mask_to_boxes((items[idx][6] - t) > 0, clean=True)
                    tp, fp, fn, _, _ = metrics.match_miss_false_alarm(pred_boxes_i, gt_boxes_i, thresh=0.3)
                    total_tp += tp
                    total_fp += fp
                    total_fn += fn
                miss = total_fn / (total_tp + total_fn) if (total_tp + total_fn) else 0.
                fa = total_fp / (total_tp + total_fp) if (total_tp + total_fp) else 0.
                score = 2 * miss + fa
                if best_score is None or score < best_score:
                    best_score, best_thr = score, float(t)
            pix_thr = best_thr
            LOGGER.info(f"final_test: pixel threshold (minrisk) = {pix_thr:.4f}, "
                        f"2*miss+false_alarm = {best_score:.4f} ({len(candidates)} 個候選門檻)")
        else:
            pix_thr = metrics.search_best_pixel_threshold(segmentations, masks_gt, criterion="f1")["threshold"]
            LOGGER.info(f"final_test: pixel threshold (f1) = {pix_thr:.4f} "
                        f"(test_threshold={test_threshold:.4f} 是 image-level 的，這裡不能共用)")
        pix_thrs = [pix_thr] * len(items)

        # ── Detection AP@IoU0.5: 標準物件偵測評估方式，整個 group 一個數字，
        #    不是單張圖的分數。每個 predict box 的信心值 = 該框範圍內 segmentation
        #    分數的最大值，所有圖的框依信心值排序後配對 GT (同一張圖內 IoU>=0.5
        #    才算命中，一個 GT 只能配一次)，算 precision-recall 曲線下面積。
        predictions_for_ap = []
        gts_per_image = {}
        preds_per_image = {}
        for idx, it in enumerate(items):
            seg_i = it[6]
            gt_boxes_i = gt_boxes_per_item[idx]
            gts_per_image[idx] = gt_boxes_i
            pred_boxes_i = mask_to_boxes((seg_i - pix_thrs[idx]) > 0, clean=True)
            preds_per_image[idx] = pred_boxes_i
            for (x, y, w, h) in pred_boxes_i:
                conf = float(seg_i[y:y + h, x:x + w].max()) if w > 0 and h > 0 else 0.
                predictions_for_ap.append((idx, (x, y, w, h), conf))
        detection_ap = metrics.compute_detection_ap(predictions_for_ap, gts_per_image, iou_thresh=0.5)
        LOGGER.info(f"final_test: Detection AP@0.5 = {detection_ap:.4f} "
                    f"({len(predictions_for_ap)} 個 predict box, "
                    f"{sum(len(v) for v in gts_per_image.values())} 個 GT box)")

        # ── Miss rate / False alarm rate: 寬鬆、雙向 coverage>=0.5 判準，
        #    只要瑕疵有被圈到就算數 (不用框得剛好)，但框太大配不上一樣算錯。
        #    群組層級池化 (跟 AP@0.5 同一種聚合層級)，不是逐圖平均。
        #    同一輪順便畫「重疊圖」(左 GT 綠框、右 predict 橘框，predict 已經
        #    套用跟計數一樣的合併邏輯) 存到 final_output/，只對有 GT 的圖畫，
        #    test/good 沒有東西可比較。
        final_output_dir = Path(self.results_path) / "final_output" / group_id
        shutil.rmtree(final_output_dir, ignore_errors=True)
        final_output_dir.mkdir(parents=True, exist_ok=True)

        total_tp = total_fp = total_fn = 0
        n_final_output = 0
        for idx in gts_per_image:
            gt_boxes_i = gts_per_image[idx]
            if not gt_boxes_i:
                continue
            tp_i, fp_i, fn_i, matched_i, fp_boxes_i = metrics.match_miss_false_alarm(
                preds_per_image[idx], gt_boxes_i, thresh=0.3)
            total_tp += tp_i
            total_fp += fp_i
            total_fn += fn_i

            src_tag, _, stem, _, orig, _, _, _ = items[idx]
            gt_panel = draw_boxes(orig, gt_boxes_i, GREEN)
            pred_panel = draw_boxes(orig, matched_i + fp_boxes_i, ORANGE)
            panel = np.hstack([gt_panel, pred_panel])
            cv2.imwrite(str(final_output_dir / f"{src_tag}{stem}.png"), panel)
            n_final_output += 1

        LOGGER.info(f"final_test: {n_final_output} 張重疊圖 (左=GT 綠框，右=predict 橘框已合併) "
                    f"存到 {final_output_dir}")
        miss_rate = total_fn / (total_tp + total_fn) if (total_tp + total_fn) else 0.
        false_alarm_rate = total_fp / (total_tp + total_fp) if (total_tp + total_fp) else 0.
        LOGGER.info(f"final_test: Miss Rate = {miss_rate:.4f}, False Alarm Rate = {false_alarm_rate:.4f} "
                    f"(coverage>=0.5 雙向判準, tp={total_tp} fp={total_fp} fn={total_fn})")

        extra_metrics = {
            "I-AUROC": img_ret["auroc"],
            "P-AUROC": pix_ret["auroc"],
            "P-PRO": pixel_pro,
            "AP@0.5(bbox)": detection_ap,
            "Miss Rate": miss_rate,
            "False Alarm": false_alarm_rate,
        }

        report_txt = pred_dir / "report.txt"
        utils.write_eval_report(cls, group_id, img_paths, labels_gt, scores, test_threshold,
                                str(report_txt), single_class=False, extra_metrics=extra_metrics,
                                pixel_threshold=pix_thr)
        LOGGER.info(f"final_test report -> {report_txt} (threshold={test_threshold:.4f}, "
                    f"thr_mode={self.thr_mode})")

        pred_csv = pred_dir / "predictions.csv"
        with open(pred_csv, mode='w', newline='') as f:
            w = csv.writer(f)
            w.writerow(["image_path", "source", "label", "label_name", "score", "pred", "correct"])
            for it, sc in zip(items, scores):
                lab = it[1]
                pred = int(sc >= test_threshold)
                w.writerow([it[3], it[0], lab, "fake" if lab else "good", f"{float(sc):.6f}",
                            pred, int(pred == lab)])
        LOGGER.info(f"final_test predictions -> {pred_csv}")

        cm_png = pred_dir / "confusion_matrix.png"
        utils.plot_confusion_matrix(cls, group_id, str(cm_png))
        LOGGER.info(f"final_test confusion matrix -> {cm_png}")

        seg_min, seg_max = float(segmentations.min()), float(segmentations.max())

        def six_panel(orig, seg, gt_arr, thr):
            pred_bin = (seg - thr) > 0
            gt_bin = gt_arr > 0
            pred_boxes = mask_to_boxes(pred_bin, clean=True)
            gt_boxes = mask_to_boxes(gt_bin, clean=False)

            gt_panel = draw_boxes(orig, gt_boxes, GREEN)
            pred_panel = draw_boxes(orig, pred_boxes, RED)
            compare_panel = draw_boxes(draw_boxes(orig, gt_boxes, GREEN), pred_boxes, RED)

            seg_norm = (seg - seg_min) / (seg_max - seg_min + 1e-8)
            heat = cv2.applyColorMap((seg_norm * 255).astype('uint8'), cv2.COLORMAP_JET)
            heat_overlay = cv2.addWeighted(orig.astype('uint8'), 0.6, heat, 0.4, 0)

            cell = (256, 256)
            row1 = np.hstack([cv2.resize(orig, cell), cv2.resize(gt_panel, cell)])
            row2 = np.hstack([cv2.resize(pred_panel, cell), cv2.resize(compare_panel, cell)])
            row3 = np.hstack([cv2.resize(heat, cell), cv2.resize(heat_overlay, cell)])
            return np.vstack([row1, row2, row3])

        preds = (scores >= test_threshold).astype(int)

        if save_visualizations:
            shutil.rmtree(viz_dir, ignore_errors=True)
            viz_dir.mkdir(parents=True, exist_ok=True)
            for it, pred, thr in zip(items, preds, pix_thrs):
                src_tag, lab, stem, _, orig, _, seg, gt_arr = it
                ok_ng = "OK" if pred == lab else "NG"
                panel = six_panel(orig, seg, gt_arr, thr)
                cv2.imwrite(str(viz_dir / f"{ok_ng}_{src_tag}{stem}.png"), panel)
            LOGGER.info(f"final_test: {len(items)} 張圖存到 {viz_dir}")

        # ── 判斷錯誤的圖另存一份，方便不用在 visualize_all 裡大海撈針 ──
        shutil.rmtree(wrong_dir, ignore_errors=True)
        wrong_dir.mkdir(parents=True, exist_ok=True)
        n_wrong = 0
        for it, pred, thr in zip(items, preds, pix_thrs):
            src_tag, lab, stem, _, orig, _, seg, gt_arr = it
            if pred == lab:
                continue
            n_wrong += 1
            panel = six_panel(orig, seg, gt_arr, thr)
            cv2.imwrite(str(wrong_dir / f"NG_{src_tag}{stem}.png"), panel)
        LOGGER.info(f"final_test: {n_wrong} 張判錯的圖存到 {wrong_dir}")

        # ── 「框得最好」的 10 張:2*miss_rate + false_alarm 最低 (分數越低越好)──
        # 跟 group 層級的 Miss Rate/False Alarm 同一套 match_miss_false_alarm
        # 判準，只是這裡是單張圖各自算，不是池化。直接重用前面已經算好的
        # gts_per_image/preds_per_image (用最終 pix_thr 切出來的框)，不重算。
        box_scores = []
        for idx in range(len(items)):
            gt_boxes_i = gts_per_image[idx]
            if not gt_boxes_i:
                box_scores.append(None)
                continue
            tp, fp, fn, matched_i, fp_boxes_i = metrics.match_miss_false_alarm(
                preds_per_image[idx], gt_boxes_i, thresh=0.3)
            miss = fn / (tp + fn) if (tp + fn) else 0.
            fa = fp / (tp + fp) if (tp + fp) else 0.
            score = 2 * miss + fa
            box_scores.append((score, miss, fa, tp, fp, fn, len(gt_boxes_i), len(preds_per_image[idx])))

        valid = sorted([(i, v) for i, v in enumerate(box_scores) if v is not None],
                       key=lambda x: x[1][0])  # 分數越低(漏檢+誤報越少)排越前面
        top10 = valid[:10]

        # 統一收在 results_path/bbox_top10/<group_id>/，不是每個 group
        # 各自散在 analyze results/<group_id>/ 底下，方便一次瀏覽所有 group 的結果。
        bbox_dir = Path(self.results_path) / "bbox_top10" / group_id
        shutil.rmtree(bbox_dir, ignore_errors=True)
        bbox_dir.mkdir(parents=True, exist_ok=True)
        top10_lines = []
        for rank, (i, (score, miss, fa, tp, fp, fn, n_gt, n_pred)) in enumerate(top10, 1):
            src_tag, lab, stem, _, orig, _, seg, gt_arr = items[i]
            panel = six_panel(orig, seg, gt_arr, pix_thrs[i])
            cv2.imwrite(str(bbox_dir / f"{rank}_{score:.3f}_{src_tag}{stem}.png"), panel)
            top10_lines.append(f"  {rank}. {src_tag}{stem}   2*miss+FA = {score:.4f}  "
                               f"(miss={miss:.2f} fa={fa:.2f}, GT {n_gt} 框 / predict {n_pred} 框 / 配對成功 {tp})")
        LOGGER.info(f"final_test: top10 bbox (2*miss+false_alarm 最低) 圖存到 {bbox_dir}")

        with open(report_txt, "a") as f:
            f.write("\n" + "-" * 72 + "\n")
            f.write(f"TOP 10 BBOX (2*miss_rate+false_alarm 最低，共 {len(valid)} 張有 GT 可比較)\n")
            f.write("-" * 72 + "\n")
            if top10_lines:
                mean_score = sum(v[0] for _, v in valid) / len(valid)
                f.write(f"mean 2*miss+FA (全部有 GT 的圖) : {mean_score:.4f}  "
                        f"(單張圖各自算，漏檢的懲罰是誤報的兩倍)\n\n")
                f.write("\n".join(top10_lines) + "\n")
            else:
                f.write("  (這個 group 沒有帶 GT 的圖，無法算)\n")

        return {"cls": cls, **extra_metrics, "n": len(items)}

    def _evaluate(self, images, scores, segmentations, labels_gt, masks_gt, name, path='training', img_paths=None):
        scores = np.squeeze(np.array(scores))

        # test set 只有單一類別 (例如手上完全沒有假鞋樣本) 時無法定義 AUROC，
        # 這時只輸出每張圖的 score 與 heatmap，不產生會誤導人的指標。
        self.single_class_test = len(np.unique(np.asarray(labels_gt).astype(int))) < 2
        if self.single_class_test:
            image_auroc = image_ap = 0.
        else:
            image_scores = metrics.compute_imagewise_retrieval_metrics(scores, labels_gt, path)
            image_auroc = image_scores["auroc"]
            image_ap = image_scores["ap"]

        # ── image-level 分類正確率 ────────────────────────────────
        # discriminator 是 sigmoid 輸出且以 BCE (正常->0, 異常->1) 訓練，
        # 門檻: thr_mode='fixed' 時就是 dsc_margin，'percentile' 時是訓練集校準出來的值，
        # 'oracle_f1'/'oracle_acc' 時直接拿這次 test (驗證集) 的分數搜尋讓 F1/accuracy
        # 最大的門檻。
        # 注意: oracle_f1/oracle_acc 用了 test 標籤去挑門檻，是樂觀上界，不是可部署的
        # 校準方式，只適合「就是要在這批固定的 test 圖上報最好的數字」這種用途
        # (常見於論文報表)。這裡的 test 是驗證集 (1:1 平衡)，算出來的 self.threshold
        # 只給 fixed/percentile 模式的 final_test() 沿用；oracle_f1/oracle_acc 模式
        # final_test() 會在展開後的完整 test 上重新搜一次，不是沿用這裡的值。
        if self.thr_mode == "oracle_f1" and not self.single_class_test:
            self.threshold = metrics.search_best_threshold(scores, labels_gt, criterion="f1")["threshold"]
        elif self.thr_mode == "oracle_acc" and not self.single_class_test:
            self.threshold = metrics.search_best_threshold(scores, labels_gt, criterion="acc")["threshold"]

        cls = metrics.compute_classification_metrics(scores, labels_gt, threshold=self.threshold)
        if self.single_class_test:
            cls["best_threshold"] = cls["best_acc"] = cls["best_balanced_acc"] = 0.
            cls["best_f1"] = cls["best_f1_threshold"] = 0.
        else:
            # 兩組獨立搜尋: bacc 最佳門檻和 f1 最佳門檻通常不是同一個。
            cls_best = metrics.search_best_threshold(scores, labels_gt, criterion="balanced_acc")
            cls["best_threshold"] = cls_best["threshold"]
            cls["best_acc"] = cls_best["acc"]
            cls["best_balanced_acc"] = cls_best["balanced_acc"]

            cls_best_f1 = metrics.search_best_threshold(scores, labels_gt, criterion="f1")
            cls["best_f1"] = cls_best_f1["f1"]
            cls["best_f1_threshold"] = cls_best_f1["threshold"]

        # 逐張影像的預測結果 (只在最終 eval 時輸出，訓練中每個 epoch 寫會太吵)
        # 注意: 這裡的 img_paths/labels_gt 來自 1:1 平衡的 test/good vs test/defect
        # (驗證集，用來校準門檻/挑 best ckpt)，不是最終展開 other_fake 的 test。
        # 所以檔名一律加 val_ 前綴，跟 final_test() 產生的正式 report 分開放，
        # 但共用同一個 <group_id> 資料夾，不要讓 analyze results 底下散一堆檔案。
        if path == 'eval' and img_paths is not None:
            group_id = name.split("_", 1)[1] if "_" in name else name
            pred_dir = os.path.join(self.results_path, "analyze results", group_id)
            os.makedirs(pred_dir, exist_ok=True)
            pred_csv = os.path.join(pred_dir, "val_predictions.csv")
            with open(pred_csv, mode='w', newline='') as f:
                w = csv.writer(f)
                w.writerow(["image_path", "label", "label_name", "score", "pred", "correct"])
                for p, lab, sc in zip(img_paths, labels_gt, np.asarray(scores).ravel()):
                    pred = int(sc >= self.threshold)
                    w.writerow([p, int(lab), "fake" if lab else "good", f"{float(sc):.6f}",
                                pred, int(pred == int(lab))])
            LOGGER.info(f"Per-image predictions written to {pred_csv}")

            # 人看的報告 (含混淆矩陣與逐張對錯，判錯的排最前面)
            report_txt = os.path.join(pred_dir, "val_report.txt")
            utils.write_eval_report(cls, name, img_paths, labels_gt,
                                    np.asarray(scores).ravel(), self.threshold,
                                    report_txt, single_class=self.single_class_test)
            LOGGER.info(f"Evaluation report written to {report_txt}")

            if not self.single_class_test:
                cm_png = os.path.join(pred_dir, "val_confusion_matrix.png")
                utils.plot_confusion_matrix(cls, name, cm_png)
                LOGGER.info(f"Confusion matrix written to {cm_png}")

        segmentations = np.array(segmentations)

        # 沒有 pixel-level ground truth 時 (image-level 二元分類模式)，跳過所有 pixel 指標。
        # pixel_auroc 回傳 0. 會讓 trainer 的 best-ckpt 判準自動退化成只看 image AUROC。
        masks_gt = np.array(masks_gt)
        has_pixel_gt = masks_gt.size > 0 and masks_gt.max() > masks_gt.min()

        if has_pixel_gt:
            pixel_scores = metrics.compute_pixelwise_retrieval_metrics(segmentations, masks_gt, path)
            pixel_auroc = pixel_scores["auroc"]
            pixel_ap = pixel_scores["ap"]
            if path == 'eval':
                try:
                    # PRO 要對 200 個門檻各做一次連通元件標記，成本隨影像數線性成長。
                    # test set 很大時抽樣一部分影像來估計，否則單這一步就要跑上一小時。
                    pro_masks, pro_segs = np.squeeze(masks_gt), segmentations
                    pro_limit = 200
                    if len(pro_masks) > pro_limit:
                        sel = np.random.default_rng(0).choice(len(pro_masks), pro_limit, replace=False)
                        pro_masks, pro_segs = pro_masks[sel], pro_segs[sel]
                    pixel_pro = metrics.compute_pro(pro_masks, pro_segs)
                except:
                    pixel_pro = 0.
            else:
                pixel_pro = 0.
        else:
            pixel_auroc = pixel_ap = pixel_pro = 0.

        defects = images
        targets = masks_gt

        save_limit = min(len(defects), 50)

        # 有 mask 就照 mask 分 NG/OK，沒有就用 image-level 標籤
        if has_pixel_gt:
            ng_indices = [i for i, target in enumerate(targets) if target.sum() > 0]
            ok_indices = [i for i, target in enumerate(targets) if target.sum() == 0]
        else:
            ng_indices = [i for i, lab in enumerate(labels_gt) if lab]
            ok_indices = [i for i, lab in enumerate(labels_gt) if not lab]

        half_limit = save_limit // 2

        ok_take = min(len(ok_indices), half_limit)
        ng_take = min(len(ng_indices), half_limit)

        if ok_take < half_limit:
            ng_take = min(len(ng_indices), save_limit - ok_take)
        elif ng_take < half_limit:
            ok_take = min(len(ok_indices), save_limit - ng_take)

        save_indices = ok_indices[:ok_take] + ng_indices[:ng_take]

        seg_min, seg_max = float(segmentations.min()), float(segmentations.max())

        def make_panel(orig_idx):
            """組出 原圖 | (GT mask) | heatmap | overlay 的並排圖。"""
            defect = defects[orig_idx]

            seg = cv2.resize(segmentations[orig_idx].astype(np.float32), (defect.shape[1], defect.shape[0]))
            # 對整個 test set 用同一組 min/max 正規化，heatmap 之間才可互相比較
            seg_norm = (seg - seg_min) / (seg_max - seg_min + 1e-8)
            heat = cv2.applyColorMap((seg_norm * 255).astype('uint8'), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(defect.astype('uint8'), 0.6, heat, 0.4, 0)

            if has_pixel_gt:
                target_mask = targets[orig_idx].astype(np.uint8)
                if target_mask.shape[0] == 1:
                    target_mask = target_mask.transpose([1, 2, 0])
                    target_mask = np.repeat(target_mask, 3, axis=-1)
                panels = [defect, target_mask * 255, heat, overlay]
            else:
                panels = [defect, heat, overlay]

            return cv2.resize(np.hstack(panels), (256 * len(panels), 256))

        # path='eval' (驗證集最終校驗) 的樣本圖跟 wrong 都收進 <group_id> 資料夾，
        # 加 val_ 前綴跟 final_test() 的正式報告分開；path='training' (訓練中每個
        # epoch 抽查) 維持原本 results_path/training/<name>/ 不變。
        if path == 'eval' and img_paths is not None:
            full_path = os.path.join(pred_dir, "val_samples") + '/'
        else:
            full_path = os.path.join(self.results_path, path, name) + '/'
        utils.del_remake_dir(full_path, del_flag=False)

        for idx, orig_idx in enumerate(save_indices):
            label_str = "NG" if orig_idx in ng_indices else "OK"
            cv2.imwrite(full_path + str(idx + 1).zfill(3) + f'_{label_str}_img{orig_idx}.png',
                        make_panel(orig_idx))

        # ── 判斷錯誤的影像另存一份 ────────────────────────────────
        # FN = 有瑕疵卻被判成正常 (漏檢，通常是最致命的那種錯)
        # FP = 正常卻被判成有瑕疵 (誤報)
        if path == 'eval' and not self.single_class_test and img_paths is not None:
            preds = (np.asarray(scores).ravel() >= self.threshold).astype(int)
            labels_arr = np.asarray(labels_gt).astype(int)
            score_arr = np.asarray(scores).ravel()

            # 漏檢按分數由低到高 (錯得最離譜的排前面)，誤報按分數由高到低
            fn_idx = sorted(np.where((labels_arr == 1) & (preds == 0))[0], key=lambda i: score_arr[i])
            fp_idx = sorted(np.where((labels_arr == 0) & (preds == 1))[0], key=lambda i: -score_arr[i])

            wrong_dir = os.path.join(pred_dir, "val_wrong")
            shutil.rmtree(wrong_dir, ignore_errors=True)
            os.makedirs(wrong_dir, exist_ok=True)

            wrong_limit = 100
            for tag, idx_list in (("FN_defect_as_good", fn_idx), ("FP_good_as_defect", fp_idx)):
                for rank, orig_idx in enumerate(idx_list[:wrong_limit], 1):
                    stem = os.path.splitext(os.path.basename(str(img_paths[orig_idx])))[0]
                    fname = f"{tag}_r{rank:03d}_s{score_arr[orig_idx]:.4f}_{stem}.png"
                    cv2.imwrite(os.path.join(wrong_dir, fname), make_panel(orig_idx))

            LOGGER.info(f"Wrong predictions: {len(fn_idx)} FN (瑕疵判成正常), "
                        f"{len(fp_idx)} FP (正常判成瑕疵) -> {wrong_dir}"
                        + (f"  [每類最多存 {wrong_limit} 張]"
                           if max(len(fn_idx), len(fp_idx)) > wrong_limit else ""))

        return image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro, cls

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

        return images, scores, masks, labels_gt, masks_gt, img_paths

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
