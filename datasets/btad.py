from torchvision import transforms
from enum import Enum

import PIL
import torch
import os

_CLASSNAMES = [
    "01", # product_1
    "02", # product_2
    "03", # product_3
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

class BTADDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            source,
            classname='01',
            resize=288,
            imagesize=288,
            split=DatasetSplit.TRAIN,
            **kwargs,
    ):
        super().__init__()
        self.source = source
        self.split = split
        self.resize = resize
        self.imgsize = imagesize
        self.classname = classname

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
        
        # 1. 讀取原始影像
        image_pil = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image_pil)

        # 2. 讀取 Ground Truth Mask
        if self.split != DatasetSplit.TRAIN and mask_path is not None:
            if os.path.exists(mask_path):
                mask_gt = PIL.Image.open(mask_path).convert('L') # 轉為灰階
                mask_gt = self.transform_mask(mask_gt)
                # 將遮罩二值化為 0 或 1
                mask_gt = torch.where(mask_gt > 0, 1.0, 0.0)
            else:
                # 若路徑存在但檔案遺失，回傳全黑遮罩
                mask_gt = torch.zeros([1, *image.size()[1:]])
        else:
            # 訓練集或無異常時，回傳全黑遮罩
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
        """
        BTAD 檔案結構邏輯:
        Train: source/train/img/01_ok_xxx.bmp
        Test:  source/test/img/01_ko_xxx.bmp (異常) 或 01_ok_xxx.bmp (正常)
        GT:    source/test/mask/01_ko_xxx.png
        """
        imgpaths_per_class = {}
        
        split_name = "train" if self.split == DatasetSplit.TRAIN else "test"
        
        # 定義影像與遮罩的資料夾路徑
        img_dir = os.path.join(self.source, split_name, "img")
        
        gt_dir = os.path.join(self.source, split_name, "mask") if split_name == "test" else None

        imgpaths_per_class[self.classname] = {}
        data_to_iterate = []

        if not os.path.exists(img_dir):
            return {}, []

        all_files = sorted(os.listdir(img_dir))
        
        # 篩選屬於當前類別 (classname, 如 '01') 的檔案
        class_files = [f for f in all_files if f.startswith(self.classname) and f.endswith(('.bmp', '.png', '.jpg'))]

        for file_name in class_files:
            image_path = os.path.join(img_dir, file_name)
            
            # 判斷是否為異常樣本
            if "_ko_" in file_name:
                anomaly_type = "bad"
            else:
                anomaly_type = "good"

            # 訓練集排除異常樣本
            if self.split == DatasetSplit.TRAIN and anomaly_type == "bad":
                continue

            mask_path = None
            if self.split == DatasetSplit.TEST and anomaly_type == "bad":
                mask_name = os.path.splitext(file_name)[0] + ".png"
                if gt_dir:
                    mask_path = os.path.join(gt_dir, mask_name)
            
            data_to_iterate.append([self.classname, anomaly_type, image_path, mask_path])
            if anomaly_type not in imgpaths_per_class[self.classname]:
                imgpaths_per_class[self.classname][anomaly_type] = []
            imgpaths_per_class[self.classname][anomaly_type].append(image_path)

        return imgpaths_per_class, data_to_iterate