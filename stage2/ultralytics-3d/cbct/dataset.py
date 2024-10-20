import pickle
import time

import nrrd
import torch
from cbct.configs import Config
from torch.utils.data import Dataset
from monai.data import CacheDataset, CacheNTransDataset, SmartCacheDataset, PersistentDataset
from trimesh.path.creation import box_outline
import os
import numpy as np
import shutil
import cv2
import imgaug as ia
import trimesh
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    EnsureType,
    RandAffined,
    Resized,
    Padd,
    Pad,
    SpatialPadd,
    DivisiblePadd,
    CenterSpatialCropd,
    RandSpatialCropSamplesd,
    RandAxisFlipd,
    RandGaussianSmoothd,
    EnsureTyped,
    RandScaleIntensityD,
)
from cbct.utils.common import vol_to_mesh, affine_and_clip_bbox, extract_bbox_from_mask, vis_mask

TIDS = [
    11, 12, 13, 14, 15, 16, 17, 18,
    21, 22, 23, 24, 25, 26, 27, 28,
    31, 32, 33, 34, 35, 36, 37, 38,
    41, 42, 43, 44, 45, 46, 47, 48,
]

REV_TIDS = [
    21, 22, 23, 24, 25, 26, 27, 28,
    11, 12, 13, 14, 15, 16, 17, 18,
    41, 42, 43, 44, 45, 46, 47, 48,
    31, 32, 33, 34, 35, 36, 37, 38,
]

maps = np.zeros((100,))
for i, tid in enumerate(TIDS, start=1):
    maps[tid] = i

rev_maps = np.zeros((100,))
for i, tid in enumerate(REV_TIDS, start=1):
    rev_maps[tid] = i


class OnlineDataset(Dataset):
    def __init__(self, cfg: Config, is_training, show=False):
        super(OnlineDataset, self).__init__()

        self.cfg = cfg
        self.is_training = is_training
        self.list_ids = self.init_list_ids()

        self.show = show

        data_root = cfg.DATA_ROOT

        list_ids = []
        for sub in [
            'sts24',
        ]:
            sub_dir = os.path.join(data_root, sub)
            list_ids += [os.path.join(data_root, sub, f) for f in sorted(os.listdir(sub_dir)) if
                         f.endswith('.seg.nrrd')]
        if self.is_training:
            self.list_ids = list_ids[:-6]
        else:
            self.list_ids = list_ids[-6:]

        self.info_list = {
            i: None for i in range(len(self.list_ids))
        }
        self.yolo_type = cfg.TRAIN_TYPE

        self.spatial_size = cfg.SPATIAL_SIZE

    def init_list_ids(self):
        train_ids, val_ids = self.split_dataset()
        if self.is_training:
            ids = train_ids
        else:
            ids = val_ids
        return ids

    def __len__(self):
        return len(self.list_ids)

    def __getitem__(self, index):
        file_name = self.list_ids[index]


        start = time.time()
        mask, _ = nrrd.read(file_name)
        mask = mask.astype(np.uint8)
        image, _ = nrrd.read(file_name.replace('.seg.nrrd', '.nrrd'))
        # print(f'time used in {time.time()-start:.2f}s')

        # clean mask
        uniq, counts = np.unique(mask, return_counts=True)
        for u, c in zip(uniq, counts):
            if c < 1000:
                mask[mask == u] = 0

        flip = False
        if self.is_training:
            if np.random.random() < .5:
                flip = True
                image = np.flip(image, [0])
                mask = np.flip(mask, [0])

        # print(flip, np.array(label.shape) / label.shape[0])
        # print(flip)
        if flip:
            mask = rev_maps[mask]
        else:
            mask = maps[mask]

        image = np.ascontiguousarray(image)
        mask = np.ascontiguousarray(mask)

        mask = torch.tensor(mask)[None]
        image = torch.tensor(image)[None]

        # if isinstance(batch['image'], torch.M):
        if hasattr(image, 'affine'):
            image.affine = torch.eye(4)

        # print(image.shape)
        amin = -1000
        amax = 3000
        image = (image.to(torch.float32) - amin) / (amax - amin)
        image = torch.clip(image, -1., 2.)

        if self.is_training:
            # amin = np.random.randint(-1000, 0)
            # amax = np.random.randint(2000, 3500)
            #
            # if np.random.random() < 0.5:
            #     image = (image.to(torch.float32) - amin) / (amax - amin)
            #     image = torch.clip(image, 0., 1.)
            # else:
            #     image = (image.to(torch.float32) - amin) / (amax - amin) * 255
            #     image = torch.clip(image, 0., 255.)
            #     image = image.to(torch.uint8).to(torch.float32) / 255

            # print(image.shape)
            a, b, c = image.shape[1:]
            if np.random.random() < .3:
                r = np.random.random() * 0.2 + 0.3
                clip = int(a * r)
                if np.random.random() < .5:
                    crop_image = image[:, :clip]
                    crop_mask = mask[:, :clip]
                else:
                    crop_image = image[:, -clip:]
                    crop_mask = mask[:, -clip:]

                if crop_mask.sum() != 0:
                    image = crop_image
                    mask = crop_mask

        # mask = mask * 0
        transform_dict = {
            'image': image,
            'label': mask,
        }
        # if self.is_training:
        #     transformed_batch = self.train_transform(transform_dict)
        # else:
        #     transformed_batch = self.val_transform(transform_dict)

        # if isinstance(transformed_batch, list):
        #     transformed_batch = transformed_batch[0]

        # print(image.shape)
        unique, counts = torch.unique(mask, return_counts=True)
        if self.show:
            vis_mask(mask[0])
        return {
            'image': image,
            'mask': mask,
            'unique': unique,
            'counts': counts,
            'path': file_name,
        }

    def show_3d(self, batch):
        if isinstance(batch, list):
            vis_batch = batch[0]
        else:
            vis_batch = batch

        image = vis_batch['image'][0]
        if isinstance(image, torch.Tensor):
            image = image.numpy()

        label = vis_batch['mask']
        if isinstance(label, torch.Tensor):
            label = label.numpy()

        if self.cfg.MODE == 'multilabel':
            c, h, w, z = label.shape
            bg = np.zeros((1, h, w, z), dtype=label.dtype)
            # label = np.argmax(np.concatenate([bg, label], axis=0), axis=0)
            label = label[0]
            label[label != 0] = label[label != 0] % 20 + 1
        else:
            label = label[0]

        y, x, z = image.shape
        for i in range(z):
            frame = np.tile(image[..., i:i + 1], (1, 1, 3))
            frame_label = label[..., i].astype(np.uint8)

            frame = (frame * 255).astype(np.uint8)
            segmaps = ia.SegmentationMapsOnImage(frame_label, frame_label.shape)
            # cv2.imshow('img', [..., i])
            cv2.imshow('vis', segmaps.draw_on_image(frame, alpha=0.5)[0][..., ::-1])
            cv2.waitKey()

    def split_dataset(self):
        cfg = self.cfg
        mode = self.is_training

        train_ids = []
        if isinstance(cfg.TRAIN_DIR, str):
            cfg.TRAIN_DIR = [cfg.TRAIN_DIR]
        if isinstance(cfg.VAL_DIR, str):
            cfg.VAL_DIR = [cfg.VAL_DIR]

        for data_dir in cfg.TRAIN_DIR:
            sub_dir = os.path.join(cfg.DATA_ROOT, data_dir)
            for root, _, files in os.walk(sub_dir):
                train_ids.extend([os.path.join(root, f) for f in files if f.endswith(cfg.SUFFIX)])

        if cfg.VAL_DIR is None or len(cfg.VAL_DIR) == 0:
            list_ids = sorted(train_ids)
            rng = np.random.RandomState(cfg.FOLD_SEED)
            rng.shuffle(list_ids)

            fold = cfg.FOLD
            fold_num = cfg.FOLD_NUM
            step = len(list_ids) // fold_num

            val_ids = list_ids[fold * step:(fold + 1) * step]
            train_ids = list_ids[:fold * step] + list_ids[(fold + 1) * step:]
        else:
            val_ids = []
            for data_dir in cfg.VAL_DIR:
                sub_dir = os.path.join(cfg.DATA_ROOT, data_dir)
                for root, _, files in os.walk(sub_dir):
                    val_ids.extend([os.path.join(root, f) for f in files if f.endswith(cfg.SUFFIX)])

        return train_ids, val_ids


if __name__ == '__main__':
    cfg = Config()
    cfg.DATA_ROOT = ''

    train_ds = OnlineDataset(cfg, is_training=True)
    train_ds.show = True
    for i in range(len(train_ds)):
        batch = train_ds[i]
        print(batch['image'].shape, batch['mask'].shape)
