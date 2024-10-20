import pickle
import nrrd
import torch
from torch.utils.data import Dataset
import os
import numpy as np
import shutil
import cv2
import imgaug as ia
import trimesh
import skimage
import scipy.ndimage as ndimage
from scipy.ndimage import distance_transform_edt
from monai.transforms import Compose, EnsureType, AsDiscrete, Activations, RandAffined, DivisiblePadd
from monai.transforms import *
import torch.nn.functional as F
import time
import edt


def find_boundary(label, kernel_size=0, backend='torch', device='cuda'):
    kz = kernel_size

    if backend == 'numpy':
        boundary = skimage.segmentation.find_boundaries(label, mode='inner').astype(np.uint8)

        kernel = np.ones((kz, kz, kz), dtype=np.uint8)
        boundary = ndimage.binary_dilation(boundary, kernel)
    elif backend == 'torch':
        with torch.no_grad():
            label = torch.tensor(label, dtype=torch.float32, device=device)
            label = label[None, None]

            lap_kernel = torch.zeros((1, 1, 3, 3, 3), device=device)
            lap_kernel[0, 0, 1, 1, :] = 1
            lap_kernel[0, 0, 1, :, 1] = 1
            lap_kernel[0, 0, :, 1, 1] = 1
            lap_kernel[0, 0, 1, 1, 1] = -6

            dilate_kernel = torch.ones((1, 1, kz, kz, kz), device=device)

            boundary = F.conv3d(label, lap_kernel.to(label.device), padding=1)
            boundary = (boundary != 0).to(torch.float32)

            boundary = F.conv3d(boundary, dilate_kernel.to(boundary.device), padding=(kz - 1) // 2)
            boundary = (boundary > 0).to(torch.float32)

            boundary = boundary[0, 0].cpu().numpy()

    return boundary



class ToothDataset(Dataset):
    def __init__(self, list_ids, is_training, dynamic_range=False, with_sdf=False, show=False):
        super(ToothDataset, self).__init__()

        self.is_training = is_training
        self.show = show

        spatial_size = (128, 128, 128)
        max_spatial_size = 128
        self.list_ids = list_ids

        self.dynamic_range = dynamic_range
        self.with_sdf = with_sdf

        if self.is_training:
            transforms = Compose(
                [
                    # Resized(
                    #     keys=['patch', 'fine', 'coarse'],
                    #     spatial_size=max_spatial_size, size_mode='longest',
                    #     mode=('trilinear', 'nearest', 'nearest'),
                    # ),
                    SpatialPadd(
                        keys=['patch', 'fine', 'coarse'],
                        spatial_size=spatial_size,
                    ),

                    RandSpatialCropd(
                        keys=['patch', 'fine', 'coarse'],
                        roi_size=spatial_size,
                    ),

                    RandAxisFlipd(
                        keys=['patch', 'fine', 'coarse'],
                        prob=0.5, ),

                    RandRotate90d(
                        keys=['patch', 'fine', 'coarse'],
                        prob=0.5, ),

                    # user can also add other random transforms
                    RandAffined(
                        keys=['patch', 'fine', 'coarse'],
                        mode=('bilinear', 'nearest', 'nearest'),
                        prob=0.5,
                        spatial_size=spatial_size,
                        translate_range=(0.1, 0.1, 0.1),
                        rotate_range=(np.pi / 10, np.pi / 10, np.pi / 10),
                        scale_range=((0, 0.4), (0., 0.4), (0, 0.4)),
                        cache_grid=False,
                        padding_mode='zeros'
                    ),
                ]
            )
        else:
            transforms = Compose(
                [
                    # Resized(
                    #     keys=['patch', 'fine', 'coarse'],
                    #     spatial_size=max_spatial_size, size_mode='longest',
                    #     mode=('trilinear', 'nearest', 'nearest'),
                    # ),

                    SpatialPadd(
                        keys=['patch', 'fine', 'coarse'],
                        spatial_size=spatial_size,
                    ),
                    CenterSpatialCropd(
                        keys=['patch', 'fine', 'coarse'],
                        roi_size=spatial_size,
                    )
                ]
            )

        self.transform = transforms

    def __getitem__(self, index):
        file_name = self.list_ids[index]

        if file_name.endswith('pkl'):
            with open(file_name, 'rb') as f:
                batch = pickle.load(f)
                patch = batch['patch']
                coarse = batch['corase']
                fine = batch['fine']

        elif file_name.endswith('nrrd'):
            patch, _ = nrrd.read(file_name)
            coarse, _ = nrrd.read(file_name.replace('patch.nrrd', 'coarse.seg.nrrd'))
            fine, _ = nrrd.read(file_name.replace('patch.nrrd', 'fine.seg.nrrd'))

        else:
            raise NotImplementedError

        patch = torch.tensor(patch.astype(np.float32))[None]
        coarse = torch.tensor(coarse.astype(np.uint8))[None]
        fine = torch.tensor(fine.astype(np.uint8))[None]
        # print(patch.shape)

        if self.dynamic_range:
            if self.is_training:
                amin = np.random.randint(-1000, 0)
                amax = np.random.randint(2000, 3500)

                if np.random.random() < 0.5:
                    patch = (patch.to(torch.float32) - amin) / (amax - amin)
                    patch = torch.clip(patch, 0., 1.)
                else:
                    patch = (patch.to(torch.float32) - amin) / (amax - amin) * 255
                    patch = torch.clip(patch, 0., 255.)
                    patch = patch.to(torch.uint8).to(torch.float32) / 255

            else:
                amin = -550
                amax = 2250
                patch = (patch.to(torch.float32) - amin) / (amax - amin) * 255
                patch = torch.clip(patch, 0., 255.)
                patch = patch.to(torch.uint8).to(torch.float32) / 255
        else:
            amin, amax = -1000, 3000
            patch = (patch.to(torch.float32) - amin) / (amax - amin)
            patch = torch.clip(patch, -1, 2.)

        aug_data = self.transform({
            'patch': patch,
            'fine': fine,
            'coarse': coarse
        })

        cp = (aug_data['coarse'][0].numpy() == 1).astype(np.uint8)
        fp = (aug_data['fine'][0].numpy() == 1).astype(np.uint8)
        if self.is_training:
            kz = np.random.randint(7, 15)
            if kz % 2 == 0:
                kz += 1
        else:
            kz = 9
        boundary = find_boundary(cp, kernel_size=kz, backend='torch')

        batch = {
            'patch': aug_data['patch'].as_tensor().numpy().astype(np.float32),
            'label': fp[None].astype(np.float32),
            'trimap': aug_data['coarse'][0].as_tensor().numpy().astype(int),

            'boundary': boundary.astype(np.float32)[None]
        }

        if self.with_sdf:
            posmask = fp.astype(bool)
            negmask = ~posmask
            posdis = edt.edt(posmask)
            negdis = edt.edt(negmask)

            negdis = negdis * boundary
            posdis = posdis * boundary
            sdf = negdis / (negdis.max() + 1e-7) - posdis / (posdis.max() + 1e-7)
            batch['sdf'] = sdf[None].astype(np.float32)

        # batch =
        if self.show:
            print(kz)
            self.show_3d(batch)
        return batch

    def __len__(self):
        return len(self.list_ids)

    def show_3d(self, batch):
        patch = batch['patch'][0]
        trimap = batch['trimap'].astype(np.int32)
        boundary = batch['boundary'][0]
        label = batch['label'][0]


        sdf = batch['sdf'][0] if 'sdf' in batch else None
        # print(patch.max(), patch.min())

        x, y, z = patch.shape
        for i in range(z):
            frame = np.tile(patch[..., i:i + 1], (1, 1, 3))
            frame = np.clip(frame, 0., 1.)
            frame = (frame * 255).astype(np.uint8)

            if sdf is not None:
                f_sdf = sdf[..., i].copy()
                f_sdf = (f_sdf + 1) / 2
                cv2.imshow('sdf', f_sdf)

            segmap = ia.SegmentationMapsOnImage(trimap[..., i].copy(), shape=frame.shape)
            vis_coarse = segmap.draw_on_image(frame, alpha=0.5)[0][..., ::-1]

            segmap = ia.SegmentationMapsOnImage(label[..., i].copy(), shape=frame.shape)
            vis_fine = segmap.draw_on_image(frame, alpha=0.5)[0][..., ::-1]

            f_boundary = boundary[..., i].copy()

            cv2.imshow('coarse', vis_coarse)
            cv2.imshow('fine', vis_fine)
            cv2.imshow('frame', frame)
            cv2.imshow('boudary', f_boundary)

            # cv2.imshow('seg', frame)
            # cv2.imshow('vis', segmaps.draw_on_image(frame, alpha=0.5)[0][..., ::-1])
            cv2.waitKey()

