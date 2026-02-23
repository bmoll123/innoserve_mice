"""
MVTec AD 2 資料集載入模組

MVTec AD 2 資料夾結構：
mvtec_ad_2/
├── category_name/ (共 8 個類別)
│   ├── train/
│   │   └── good/ (完全無瑕疵的正常影像)
│   ├── validation/ (驗證集 - MVTec AD 2 新增)
│   │   └── good/ (用於模型調參的正常影像)
│   ├── test_public/ (公開測試集)
│   │   ├── good/ (正常影像)
│   │   ├── bad/ (瑕疵影像，例如: 000_overexposed.png)
│   │   └── ground_truth/
│   │       └── bad/ (mask 影像，例如: 000_overexposed_mask.png)
│   ├── test_private/ (私有測試集)
│   └── test_private_mixed/ (混合私有測試集)
"""

from torchvision import transforms
from enum import Enum

import PIL
import torch
import os

_CLASSNAMES = [
    "can",
    "fabric",
    "fruit_jelly",
    "rice",
    "sheet_metal",
    "vial",
    "wallplugs",
    "walnuts",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "validation"
    TEST = "test_public"
    TEST_PRIVATE = "test_private"
    TEST_PRIVATE_MIXED = "test_private_mixed"


class MVTec2Dataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for MVTec AD 2.
    """

    def __init__(
            self,
            source,
            classname='can',
            resize=288,
            imagesize=288,
            split=DatasetSplit.TRAIN,
            **kwargs,
    ):
        """
        Args:
            source: [str]. 指向 MVTec AD 2 資料夾的路徑。
            classname: [str or None]. MVTec AD 2 類別名稱。
                       如果為 None，資料集會遍歷所有可用的影像。
            resize: [int]. 載入影像後初始調整的大小（正方形）。
            imagesize: [int]. 調整大小後進行中心裁剪的大小（正方形）。
            split: [enum-option]. 指定使用訓練集、驗證集或測試集。
        """
        super().__init__()
        self.source = source
        self.split = split
        self.resize = resize
        self.imgsize = imagesize
        self.imagesize = (3, self.imgsize, self.imgsize)
        self.classname = classname

        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

        self.transform_img = [
            transforms.Resize(self.resize),
            transforms.CenterCrop(self.imgsize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

        self.transform_mask = [
            transforms.Resize(self.resize),
            transforms.CenterCrop(self.imgsize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        if self.split == DatasetSplit.TEST and mask_path is not None:
            mask_gt = PIL.Image.open(mask_path).convert('L')
            mask_gt = self.transform_mask(mask_gt)
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
        """
        根據 MVTec AD 2 的資料夾結構獲取影像路徑。

        資料夾結構：
        - train/good/
        - validation/good/
        - test_public/good/, test_public/bad/, test_public/ground_truth/bad/
        """
        imgpaths_per_class = {}
        maskpaths_per_class = {}

        set_name = self.split.value
        classpath = os.path.join(self.source, self.classname, set_name)

        if self.split == DatasetSplit.TEST:
            maskpath = os.path.join(self.source, self.classname, set_name, "ground_truth")
        else:
            maskpath = None

        imgpaths_per_class[self.classname] = {}
        maskpaths_per_class[self.classname] = {}

        # 處理訓練集和驗證集（只有 good 資料夾）
        if self.split in [DatasetSplit.TRAIN, DatasetSplit.VAL]:
            good_path = os.path.join(classpath, "good")
            if os.path.exists(good_path):
                good_files = sorted([
                    f for f in os.listdir(good_path)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
                ])
                imgpaths_per_class[self.classname]["good"] = [
                    os.path.join(good_path, x) for x in good_files
                ]
            else:
                imgpaths_per_class[self.classname]["good"] = []
            maskpaths_per_class[self.classname]["good"] = None

        # 處理公開測試集（有 good 和 bad 資料夾）
        elif self.split == DatasetSplit.TEST:
            # 取得 classpath 下的所有子目錄（排除 ground_truth）
            if os.path.exists(classpath):
                subdirs = [
                    d for d in os.listdir(classpath)
                    if os.path.isdir(os.path.join(classpath, d)) and d != "ground_truth"
                ]
            else:
                subdirs = []

            for anomaly in subdirs:
                anomaly_path = os.path.join(classpath, anomaly)
                anomaly_files = sorted([
                    f for f in os.listdir(anomaly_path)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
                ])
                imgpaths_per_class[self.classname][anomaly] = [
                    os.path.join(anomaly_path, x) for x in anomaly_files
                ]

                # 處理 mask（只有非 good 類別才有 mask）
                if anomaly != "good" and maskpath is not None:
                    anomaly_mask_path = os.path.join(maskpath, anomaly)
                    if os.path.exists(anomaly_mask_path):
                        maskpaths_per_class[self.classname][anomaly] = []
                        for img_file in anomaly_files:
                            img_name = os.path.splitext(img_file)[0]
                            mask_file = f"{img_name}_mask.png"
                            mask_full_path = os.path.join(anomaly_mask_path, mask_file)
                            if os.path.exists(mask_full_path):
                                maskpaths_per_class[self.classname][anomaly].append(mask_full_path)
                            else:
                                maskpaths_per_class[self.classname][anomaly].append(None)
                    else:
                        maskpaths_per_class[self.classname][anomaly] = [None] * len(anomaly_files)
                else:
                    maskpaths_per_class[self.classname][anomaly] = None

        elif self.split in [DatasetSplit.TEST_PRIVATE, DatasetSplit.TEST_PRIVATE_MIXED]:
            if os.path.exists(classpath):
                items = os.listdir(classpath)
                subdirs = [d for d in items if os.path.isdir(os.path.join(classpath, d))]

                if subdirs:
                    for anomaly in subdirs:
                        anomaly_path = os.path.join(classpath, anomaly)
                        anomaly_files = sorted([
                            f for f in os.listdir(anomaly_path)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
                        ])
                        imgpaths_per_class[self.classname][anomaly] = [
                            os.path.join(anomaly_path, x) for x in anomaly_files
                        ]
                        maskpaths_per_class[self.classname][anomaly] = None
                else:
                    img_files = sorted([
                        f for f in items
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
                    ])
                    imgpaths_per_class[self.classname]["unknown"] = [
                        os.path.join(classpath, x) for x in img_files
                    ]
                    maskpaths_per_class[self.classname]["unknown"] = None

        data_to_iterate = []
        for anomaly in sorted(imgpaths_per_class[self.classname].keys()):
            img_paths = imgpaths_per_class[self.classname][anomaly]
            mask_paths = maskpaths_per_class[self.classname].get(anomaly)

            for i, image_path in enumerate(img_paths):
                data_tuple = [self.classname, anomaly, image_path]

                if mask_paths is not None and isinstance(mask_paths, list) and i < len(mask_paths):
                    data_tuple.append(mask_paths[i])
                else:
                    data_tuple.append(None)

                data_to_iterate.append(data_tuple)

        return imgpaths_per_class, data_to_iterate
