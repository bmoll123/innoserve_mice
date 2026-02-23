from torchvision import transforms
from enum import Enum

import PIL
import torch
import os

_CLASSNAMES = [
    "bracket_black",
    "bracket_brown",
    "bracket_white",
    "connector",
    "metal_plate",
    "tubes",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class MPDDDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for MPDD (Metal Parts Defect Detection).
    """

    def __init__(
            self,
            source,
            classname='bracket_black',
            resize=288,
            imagesize=288,
            split=DatasetSplit.TRAIN,
            **kwargs,
    ):
        """
        Args:
            source: [str]. 指向 MPDD 資料集的根目錄路徑。
            classname: [str or None]. MPDD 的類別名稱。
            resize: [int]. 載入圖片後的初始大小。
            imagesize: [int]. 最終裁切輸入模型的大小。
            split: [enum-option]. 資料集分割 (TRAIN 或 TEST)。
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

        if self.split != DatasetSplit.TRAIN and mask_path is not None:
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
        imgpaths_per_class = {}
        maskpaths_per_class = {}

        # 設定資料夾路徑，MPDD 結構通常為 classname/train 或 classname/test
        set_name = self.split.value if self.split != DatasetSplit.VAL else "test"
        classpath = os.path.join(self.source, self.classname, set_name)
        maskpath = os.path.join(self.source, self.classname, "ground_truth")
        
        # 讀取該 split 下的所有子資料夾 (例如 good, defect_type...)
        anomaly_types = sorted(os.listdir(classpath))

        imgpaths_per_class[self.classname] = {}
        maskpaths_per_class[self.classname] = {}

        for anomaly in anomaly_types:
            anomaly_path = os.path.join(classpath, anomaly)
            anomaly_files = sorted([f for f in os.listdir(anomaly_path) if f.endswith(('.png', '.jpg', '.bmp', '.tif'))])
            imgpaths_per_class[self.classname][anomaly] = [os.path.join(anomaly_path, x) for x in anomaly_files]

            if self.split != DatasetSplit.TRAIN and anomaly != "good":
                anomaly_mask_path = os.path.join(maskpath, anomaly)
                if os.path.isdir(anomaly_mask_path):
                    anomaly_mask_files = sorted([f for f in os.listdir(anomaly_mask_path) if f.endswith(('.png', '.jpg', '.bmp', '.tif'))])
                    maskpaths_per_class[self.classname][anomaly] = [os.path.join(anomaly_mask_path, x) for x in anomaly_mask_files]
                else:
                    # 若找不到對應 mask 資料夾，則設為 None (防止 crash)
                    maskpaths_per_class[self.classname][anomaly] = None
            else:
                maskpaths_per_class[self.classname]["good"] = None

        data_to_iterate = []
        for anomaly in sorted(imgpaths_per_class[self.classname].keys()):
            # 確保有 mask 列表可以索引
            has_masks = maskpaths_per_class[self.classname].get(anomaly) is not None
            
            for i, image_path in enumerate(imgpaths_per_class[self.classname][anomaly]):
                data_tuple = [self.classname, anomaly, image_path]
                if self.split != DatasetSplit.TRAIN and anomaly != "good" and has_masks:
                    # 嘗試取得對應的 mask
                    if i < len(maskpaths_per_class[self.classname][anomaly]):
                        data_tuple.append(maskpaths_per_class[self.classname][anomaly][i])
                    else:
                        data_tuple.append(None)
                else:
                    data_tuple.append(None)
                data_to_iterate.append(data_tuple)

        return imgpaths_per_class, data_to_iterate