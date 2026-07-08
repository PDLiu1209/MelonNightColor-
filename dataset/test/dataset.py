import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import dataset.tfs as T
import h5py
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom  # 如果没用到可移除
import random

# -----------------------------
# 增强与预处理（保持入参/出参不变）
# -----------------------------
class SegmentationPresetTrain:
    def __init__(self, rcrop_size=224, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        trans = [T.RandomResize(rcrop_size, rcrop_size)]
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob))
        trans.extend([
            T.RandomCrop(rcrop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetVal:
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), rcrop_size=224):
        self.transforms = T.Compose([
            T.Resize((rcrop_size, rcrop_size)),  # 传 tuple
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])


    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetTest:
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), rcrop_size=224):
        self.transforms = T.Compose([
            T.Resize((rcrop_size, rcrop_size)),  # 传 tuple
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])


    def __call__(self, img, target):
        return self.transforms(img, target)


# -----------------------------
# 路径与掩码辅助（新增内部工具，不改变外部接口）
# -----------------------------
def _exists(p):
    try:
        return os.path.exists(p)
    except Exception:
        return False

def _with_new_suffix(path, new_suffix):
    root, _ = os.path.splitext(path)
    return root + new_suffix

def _guess_mask_path(image_path):
    """
    同时支持两种常见结构：
    1) 经典结构: .../images/xxx.jpg -> .../masks/xxx.png (或相同后缀)
    2) ISIC2018: 
       Task1_Training_Input/ISIC_XXXX.jpg -> Task1_Training_GroundTruth/ISIC_XXXX_segmentation.png
       Task1_Validation_Input/...         -> Task1_Validation_GroundTruth/...
       Task1_Test_Input/...               -> Task1_Test_GroundTruth/...
    """
    # 优先：经典 images -> masks
    if "images" in image_path:
        mask_path = image_path.replace(os.sep + "images" + os.sep, os.sep + "masks" + os.sep)
        if _exists(mask_path):
            return mask_path
        # 尝试 jpg->png / jpeg->png
        for suf in [".png", ".bmp", ".tif", ".tiff"]:
            cand = _with_new_suffix(mask_path, suf)
            if _exists(cand):
                return cand

    # ISIC2018 规则映射
    mapping = [
        ("Task1_Training_Input", "Task1_Training_GroundTruth"),
        ("Task1_Validation_Input", "Task1_Validation_GroundTruth"),
        ("Task1_Test_Input", "Task1_Test_GroundTruth"),
    ]
    for a, b in mapping:
        if a in image_path:
            mask_path = image_path.replace(a, b)
            # ISIC 掩码命名：ISIC_XXXX.jpg -> ISIC_XXXX_segmentation.png
            base = os.path.basename(mask_path)
            stem, _ = os.path.splitext(base)
            seg = stem + "_segmentation.png"
            mask_path = os.path.join(os.path.dirname(mask_path), seg)
            if _exists(mask_path):
                return mask_path

    # 兜底：同目录、同名仅后缀变化
    dirname = os.path.dirname(image_path)
    stem, _ = os.path.splitext(os.path.basename(image_path))
    for cand in [
        os.path.join(dirname, stem + ".png"),
        os.path.join(dirname, stem + "_mask.png"),
        os.path.join(dirname, stem + "_seg.png"),
        os.path.join(dirname, stem + "_segmentation.png"),
    ]:
        if _exists(cand):
            return cand

    return None  # 交由上层报错，便于发现问题


def _to_class_indices(mask_arr, txt_labels):
    """
    将任意灰度掩码转为类别索引：
    - 若 txt_labels 能覆盖灰度值，使用 txt 的索引映射
    - 若检测到二值 0/255（或非零），自动映射到 {0,1}
    """
    uniques = np.unique(mask_arr)

    # 情况 A：标准的 ISIC 二值 0/255 或 0/非0
    if set(uniques.tolist()).issubset({0, 255}) or (len(uniques) <= 3 and 0 in uniques):
        bin_mask = (mask_arr > 0).astype(np.uint8)
        return bin_mask  # {0,1}

    # 情况 B：使用 txt 中列出的灰度映射（兼容你的原逻辑）
    mapped = mask_arr.copy()
    # 若灰度值未出现在 txt 里，视为前景（1）
    fg_idx = 1 if "1" in txt_labels else (txt_labels.index("255") if "255" in txt_labels else 1)
    for g in uniques:
        g_str = str(int(g))
        if g_str in txt_labels:
            mapped[mapped == g] = txt_labels.index(g_str)
        else:
            mapped[mapped == g] = fg_idx
    return mapped.astype(np.uint8)


# -----------------------------
# 通用数据集（保持入参与出参）
# -----------------------------
class MyDataset(Dataset):
    def __init__(self, imgs_path, txt_path, transform=None):
        self.imgs = [os.path.join(imgs_path, i) for i in os.listdir(imgs_path)]
        self.imgs = sorted([p for p in self.imgs if os.path.isfile(p)])
        self.transform = transform

        with open(txt_path, 'r') as f:
            self.txt = f.read().splitlines()  # 例如 ['0', '255'] 或多类

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        image_path = self.imgs[index]
        mask_path = _guess_mask_path(image_path)
        if mask_path is None or not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found for image: {image_path}")

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        mask = np.array(mask)

        # 将灰度掩码转为类别索引（自动二值化/按 txt 映射）
        mask = _to_class_indices(mask, self.txt)
        mask = Image.fromarray(mask)

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return image, mask


# -----------------------------
# NPY_datasets：自动识别 ISIC2018 目录（保持签名不变）
# -----------------------------
class NPY_datasets(Dataset):
    def __init__(self, path_Data, config, train=True):
        super(NPY_datasets, self)
        self.data = []

        # 1) 先尝试你原有的 train/val 目录结构
        if train and os.path.isdir(os.path.join(path_Data, 'train', 'images')):
            images_list = sorted(os.listdir(os.path.join(path_Data, 'train', 'images')))
            masks_list = sorted(os.listdir(os.path.join(path_Data, 'train', 'masks'))) \
                if os.path.isdir(os.path.join(path_Data, 'train', 'masks')) else images_list
            for i in range(len(images_list)):
                img_path = os.path.join(path_Data, 'train', 'images', images_list[i])
                # 若没有显式 masks 目录，用 _guess_mask_path
                if os.path.isdir(os.path.join(path_Data, 'train', 'masks')):
                    mask_path = os.path.join(path_Data, 'train', 'masks', masks_list[i])
                else:
                    mask_path = _guess_mask_path(img_path)
                self.data.append([img_path, mask_path])
            self.transformer = config.train_transformer

        elif (not train) and os.path.isdir(os.path.join(path_Data, 'val', 'images')):
            images_list = sorted(os.listdir(os.path.join(path_Data, 'val', 'images')))
            masks_list = sorted(os.listdir(os.path.join(path_Data, 'val', 'masks'))) \
                if os.path.isdir(os.path.join(path_Data, 'val', 'masks')) else images_list
            for i in range(len(images_list)):
                img_path = os.path.join(path_Data, 'val', 'images', images_list[i])
                if os.path.isdir(os.path.join(path_Data, 'val', 'masks')):
                    mask_path = os.path.join(path_Data, 'val', 'masks', masks_list[i])
                else:
                    mask_path = _guess_mask_path(img_path)
                self.data.append([img_path, mask_path])
            self.transformer = config.test_transformer

        else:
            # 2) 自动匹配 ISIC2018 的 Task1 目录
            # 训练
            if train:
                input_dir = os.path.join(path_Data, 'Task1_Training_Input')
                gt_dir = os.path.join(path_Data, 'Task1_Training_GroundTruth')
            else:
                # 验证优先；若没有验证则尝试 Test（有的发布包没有官方验证集）
                input_dir = os.path.join(path_Data, 'Task1_Validation_Input')
                gt_dir = os.path.join(path_Data, 'Task1_Validation_GroundTruth')
                if not os.path.isdir(input_dir):
                    input_dir = os.path.join(path_Data, 'Task1_Test_Input')
                    gt_dir = os.path.join(path_Data, 'Task1_Test_GroundTruth')

            if not os.path.isdir(input_dir) or not os.path.isdir(gt_dir):
                raise FileNotFoundError(
                    f"Cannot locate expected directories under {path_Data}. "
                    f"Tried standard train/val and ISIC2018 Task1 folders."
                )

            images_list = sorted([f for f in os.listdir(input_dir)
                                  if os.path.isfile(os.path.join(input_dir, f))])
            for fname in images_list:
                img_path = os.path.join(input_dir, fname)
                stem, _ = os.path.splitext(fname)
                mask_name = stem + "_segmentation.png"
                msk_path = os.path.join(gt_dir, mask_name)
                if not os.path.exists(msk_path):
                    # 兜底：使用通用推断
                    msk_path = _guess_mask_path(img_path)
                self.data.append([img_path, msk_path])

            self.transformer = config.train_transformer if train else config.test_transformer

    def __getitem__(self, indx):
        img_path, msk_path = self.data[indx]
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        if not os.path.exists(msk_path):
            raise FileNotFoundError(f"Mask not found: {msk_path}")

        img = np.array(Image.open(img_path).convert('RGB'))
        msk = np.array(Image.open(msk_path).convert('L'))

        # ISIC2018 掩码是 0/255：转为 {0,1} 并扩维到 HxWx1，保持你原先的输出格式
        if msk.max() > 1:
            msk = (msk > 0).astype(np.uint8)
        msk = np.expand_dims(msk, axis=2)  # H x W x 1

        img, msk = self.transformer((img, msk))
        return img, msk

    def __len__(self):
        return len(self.data)


# -----------------------------
# 你已有的增强函数（原样保留）
# -----------------------------
def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label
