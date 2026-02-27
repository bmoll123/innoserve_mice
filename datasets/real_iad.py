from torchvision import transforms
from enum import Enum
import PIL
import torch
import os
import random

_CLASSNAMES = [
    "audiojack", "bottle_cap", "button_battery", "end_cap", "eraser",
    "fire_hood", "mint", "mounts", "pcb", "phone_battery", "plastic_nut",
    "plastic_plug", "porcelain_doll", "regulator", "rolled_strip_base",
    "sim_card_set", "switch", "tape", "terminalblock", "toothbrush",
    "toy", "toy_brick", "transistor1", "u_block", "usb", "usb_adaptor",
    "vcpill", "wooden_beads", "woodstick", "zipper"
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

class RealIADDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for Real-IAD (Real Industrial Anomaly Detection).
    
    資料集結構假設:
    root/
    ├── class_name/
    │   ├── OK/
    │   │   └── [子資料夾]/
    │   │       └── image.jpg
    │   ├── NG/
    │   │   └── [NG類別]/
    │   │       └── [子資料夾]/
    │   │           ├── image.jpg (RGB)
    │   │           └── image.png (Mask)
    """

    def __init__(
            self,
            source,
            classname='audiojack',
            resize=288,
            imagesize=288,
            split=DatasetSplit.TRAIN,
            split_ratio=0.8,
            seed=0,
            **kwargs,
    ):
        """
        Args:
            source: [str]. Real-IAD 資料集根目錄路徑。
            classname: [str]. 類別名稱。
            resize: [int]. 圖片載入後的初始大小。
            imagesize: [int]. 圖片裁切後的最終輸入大小。
            split: [enum-option]. TRAIN 或 TEST。
            split_ratio: [float]. 將 OK 樣本劃分為訓練集的比例 (預設 0.8)。
                          因為 Real-IAD 的 OK 資料夾通常未分 Train/Test，
                          需手動切分以確保測試集包含正常樣本 (計算 AUROC 必需)。
            seed: [int]. 隨機種子，確保 Train/Test 切分的一致性。
        """
        super().__init__()
        self.source = source
        self.split = split
        self.resize = resize
        self.imgsize = imagesize
        self.imagesize = (3, self.imgsize, self.imgsize)
        self.classname = classname
        self.split_ratio = split_ratio
        self.seed = seed

        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

        self.transform_img = transforms.Compose([
            transforms.Resize(self.resize),
            transforms.CenterCrop(self.imgsize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        self.transform_mask = transforms.Compose([
            transforms.Resize(self.resize),
            transforms.CenterCrop(self.imgsize),
            transforms.ToTensor(),
        ])

    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        
        # 1. 讀取 RGB 影像
        image_pil = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image_pil)

        # 2. 處理 Mask
        if self.split != DatasetSplit.TRAIN and mask_path is not None:
            if os.path.exists(mask_path):
                mask_gt = PIL.Image.open(mask_path).convert('L')
                mask_gt = self.transform_mask(mask_gt)
                mask_gt = torch.where(mask_gt > 0.5, 1.0, 0.0) # 二值化
            else:
                mask_gt = torch.zeros([1, *image.size()[1:]])
        else:
            mask_gt = torch.zeros([1, *image.size()[1:]])

        return {
            "image": image,
            "mask_gt": mask_gt,
            "is_anomaly": int(anomaly != "good"),
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        imgpaths_per_class = {self.classname: {}}
        data_to_iterate = []

        ok_dir = os.path.join(self.source, self.classname, "OK")
        ng_dir = os.path.join(self.source, self.classname, "NG")

        # 1. 收集所有 OK 樣本 (遞迴搜尋所有子資料夾)
        ok_images = []
        if os.path.exists(ok_dir):
            for root, dirs, files in os.walk(ok_dir):
                for f in sorted(files):
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        ok_images.append(os.path.join(root, f))
        
        # 為了確保 Train/Test 切分一致，先排序再隨機洗牌
        ok_images.sort()
        random.seed(self.seed)
        random.shuffle(ok_images)

        # 根據比例切分 OK 樣本
        split_idx = int(len(ok_images) * self.split_ratio)
        train_ok_images = ok_images[:split_idx]
        test_ok_images = ok_images[split_idx:]

        # 2. 收集所有 NG 樣本 (遞迴搜尋 NG -> 類別 -> 子資料夾)
        ng_samples = [] # 格式: (anomaly_type, img_path, mask_path)
        
        if os.path.exists(ng_dir):
            # 遍歷 NG 下的第一層目錄
            anomaly_types = sorted([d for d in os.listdir(ng_dir) if os.path.isdir(os.path.join(ng_dir, d))])
            
            for anomaly_type in anomaly_types:
                type_path = os.path.join(ng_dir, anomaly_type)
                
                # 遞迴搜尋該 NG 類別下的所有子資料夾
                for root, dirs, files in os.walk(type_path):
                    for f in sorted(files):
                        # 以 mask (.png) 為基準來尋找 RGB (.jpg)
                        if f.lower().endswith('.png'):
                            mask_path = os.path.join(root, f)
                            fname_no_ext = os.path.splitext(f)[0]

                            img_path = os.path.join(root, fname_no_ext + ".jpg")

                            if not os.path.exists(img_path):
                                img_path_jpeg = os.path.join(root, fname_no_ext + ".jpeg")
                                img_path_JPG = os.path.join(root, fname_no_ext + ".JPG") # 處理大寫副檔名
                                if os.path.exists(img_path_jpeg):
                                    img_path = img_path_jpeg
                                elif os.path.exists(img_path_JPG):
                                    img_path = img_path_JPG
                                else:
                                    continue

                            ng_samples.append((anomaly_type, img_path, mask_path))

        # 3. 根據 split 構建最終列表
        if self.split == DatasetSplit.TRAIN:
            imgpaths_per_class[self.classname]["good"] = train_ok_images
            for img_path in train_ok_images:
                data_to_iterate.append([self.classname, "good", img_path, None])
                
        elif self.split == DatasetSplit.TEST or self.split == DatasetSplit.VAL:
            imgpaths_per_class[self.classname]["good"] = test_ok_images
            for img_path in test_ok_images:
                data_to_iterate.append([self.classname, "good", img_path, None])
            
            for (anomaly_type, img_path, mask_path) in ng_samples:
                if anomaly_type not in imgpaths_per_class[self.classname]:
                    imgpaths_per_class[self.classname][anomaly_type] = []
                imgpaths_per_class[self.classname][anomaly_type].append(img_path)
                
                data_to_iterate.append([self.classname, anomaly_type, img_path, mask_path])

        return imgpaths_per_class, data_to_iterate