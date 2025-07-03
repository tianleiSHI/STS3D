from os.path import dirname
import sys
import os

# sys.path.append(dirname(__file__))
file = os.path.abspath(__file__)
sys.path.append(os.path.join(dirname(dirname(file))))

# print(sys.path)
import cv2
import pytorch_lightning
import yaml
from monai.utils import set_determinism
from monai.metrics import DiceMetric
from monai.losses import DiceLoss, DiceFocalLoss, DiceCELoss, FocalLoss
from monai.inferers import sliding_window_inference
from monai.config import print_config
import torch

import shutil
import os
import glob
from cbct.configs import Config
import socket
from cbct.utils.nn import CosineAnnealingWarmupLR
from cbct.dataset import OnlineDataset
import numpy as np
from monai.transforms import Compose, EnsureType, AsDiscrete, Activations
from monai.data import list_data_collate, decollate_batch, DataLoader
import torch.nn.functional as F
from monai.networks.utils import one_hot
from torchvision.ops.focal_loss import sigmoid_focal_loss
from functools import partial
import torch.nn as nn
from monai.transforms import Compose, EnsureType, AsDiscrete, Activations, RandAffined, DivisiblePadd
from cbct.utils.common import extract_bbox_from_mask, show_3d, vis_mask
import trimesh
from typing import List, Dict, Sequence
import collections
import re
from ultralytics import YOLO
from cbct.loss import SegLoss
from ultralytics.yolo.utils import ops as ops
import sys
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
    DivisiblePadd,
    FgBgToIndicesd,
    SpatialPadd,
    Flipd,
    RandFlipd,
    RandAxisFlipd,
    RandCropd,
    RandSpatialCropd,
    RandScaleCropd,
    Resized,
    RandAffined,
    Crop
)


# torch.autograd.set_detect_anomaly(True)


def custom_data_collate(batch, transform=None):
    # augmentation
    batch = batch[0]
    batch['image'] = batch['image'][None]
    batch['mask'] = batch['mask'][None]

    collated_batch = {}
    for k in batch[0].keys():
        collated_batch['image']

    return batch


class PLModel(pytorch_lightning.LightningModule):
    def __init__(self, cfg: Config, export=False, fp16=False):
        super().__init__()

        self.cfg = cfg
        self.export = export
        if cfg.DATASET_TYPE == 'online':
            self.train_ds = OnlineDataset(cfg, is_training=True)
            self.val_ds = OnlineDataset(cfg, is_training=False)

        cfg_path = 'cbct/yolov8s-seg.yaml'
        cfg_path = 'cbct/yolov8n-seg-f4.yaml'
        # cfg_path = 'cbct/yolov8s-seg-f4-fairy.yaml'

        self.yolo = YOLO(cfg_path)
        self.model = self.yolo.model

        # self.dice_loss = DiceLoss(mode='binary', from_logits=True)
        self.dice_loss = DiceLoss(
            sigmoid=True,
        )
        # self.focal_loss = FocalLoss(reduction='none')

        self.hyp = {
            'anchor_t': 4.
        }
        self.fp16 = fp16

        self.loss_info = []
        self.seg_loss = None
        # self.loss_function = DiceFocalLoss(sigmoid=True, to_onehot_y=True)
        # spatial_size = (320, 320, 320)
        spatial_size = self.cfg.SPATIAL_SIZE
        self.transform = Compose([
            Resized(
                keys=["image", "label"],
                mode=('bilinear', 'nearest'),
                spatial_size=320,
                size_mode='longest',
            ),
            # RandCropd(
            # ),
            RandSpatialCropd(
                keys=["image", "label"],
                roi_size=(320, 288, 224),
            ),
            DivisiblePadd(
                keys=["image", "label"],
                k=32,
            ),

            # RandAffined(
            #     keys=['image', 'label'],
            #     mode=('bilinear', 'nearest'),
            #     prob=.3,
            #     spatial_size=None,
            #     translate_range=(.2, .2, .2),
            #     rotate_range=(np.pi / 15, np.pi / 15, np.pi / 15),
            #     scale_range=(0.3, 0.3, 0.3)),
        ]
        )

        self.pad_transform = Compose([
            # RandScaleCropd(
            # ),
            DivisiblePadd(
                keys=["image", "label"],
                k=32,
            ),

            # RandAffined(
            #     keys=['image', 'label'],
            #     mode=('bilinear', 'nearest'),
            #     prob=.3,
            #     spatial_size=None,
            #     translate_range=(.2, .2, .2),
            #     rotate_range=(np.pi / 15, np.pi / 15, np.pi / 15),
            #     scale_range=(0.3, 0.3, 0.3)),
        ])


        self.val_transform = Compose([
            Resized(
                keys=["image", "label"],
                mode=('bilinear', 'nearest'),
                spatial_size=320,
                size_mode='longest',
            ),
            DivisiblePadd(
                keys=["image", "label"],
                k=32,
            ),
        ])

        self.dice_metric = DiceMetric(include_background=True, reduction='none', get_not_nans=False)
        self.best_val_dice = 0
        self.best_val_epoch = 0
        self.cfg = cfg

    def forward(self, x, resize_shape=None, pad_width=None):
        return self.model(x)

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_ds, batch_size=self.cfg.BATCH_SIZE, shuffle=True,
            num_workers=4,
            # collate_fn=custom_data_collate,
            persistent_workers=False,
            drop_last=True,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_ds, batch_size=1, num_workers=1,
            persistent_workers=False,
            # collate_fn=custom_data_collate
        )
        return val_loader

    @staticmethod
    def build_optimizer(model, name='Adam', lr=0.001, momentum=0.9, decay=1e-5):
        """
        Builds an optimizer with the specified parameters and parameter groups.

        Args:
            model (nn.Module): model to optimize
            name (str): name of the optimizer to use
            lr (float): learning rate
            momentum (float): momentum
            decay (float): weight decay

        Returns:
            optimizer (torch.optim.Optimizer): the built optimizer
        """
        g = [], [], []  # optimizer parameter groups
        bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
        for v in model.modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias (no decay)
                g[2].append(v.bias)
            if isinstance(v, bn):  # weight (no decay)
                g[1].append(v.weight)
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
                g[0].append(v.weight)

        if name == 'Adam':
            optimizer = torch.optim.Adam(g[2], lr=lr, betas=(momentum, 0.999))  # adjust beta1 to momentum
        elif name == 'AdamW':
            optimizer = torch.optim.AdamW(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif name == 'RMSProp':
            optimizer = torch.optim.RMSprop(g[2], lr=lr, momentum=momentum)
        elif name == 'SGD':
            optimizer = torch.optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(f'Optimizer {name} not implemented.')

            # for name, param in model.named_parameters():
            #     if '25.cv3' not in name:
            #         param.requires_grad = False
            pass
        from ultralytics.yolo.utils import LOGGER, colorstr
        optimizer.add_param_group({'params': g[0], 'weight_decay': decay})  # add g0 with weight_decay
        optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})  # add g1 (BatchNorm2d weights)
        LOGGER.info(f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}) with parameter groups "
                    f'{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias')
        return optimizer

    def configure_optimizers(self):
        epochs = self.cfg.EPOCHS

        num_batches = len(self.train_dataloader()) / self.trainer.accumulate_grad_batches
        total_iters = num_batches * epochs

        if isinstance(self.cfg.GPUS, int):
            gpu_num = self.cfg.GPUS
        else:
            gpu_num = len(self.cfg.GPUS)

        total_iters = int(total_iters / gpu_num)
        if self.cfg.LIMIT is not None:
            total_iters = int(total_iters * self.cfg.LIMIT)

        beta2 = 0.997
        warmup_iters = min(500, total_iters // 20)
        warmup_iters = max(warmup_iters, 20)
        warmup_iters = 5
        print(f'warmup iteration number: {warmup_iters}')

        lr = self.cfg.LR
        optimizer = self.build_optimizer(self.model, lr=lr)
        scheduler = CosineAnnealingWarmupLR(optimizer, total_iters=total_iters, warmup_iters=warmup_iters)
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step', 'frequency': 1}]

    def process_input_batch(self, batch, mode='train', show=False):
        images, masks = batch["image"], batch["mask"]
        _, _, a, b, c = images.shape
        shape = np.array([a, b, c])
        a, b, c = np.ceil(shape / 32) * 32

        if mode == 'train':
            # if True:
            transform = self.transform
            spatial_ratio = np.random.random(3, ) * 0.3 + 0.7
            # spatial_ratio = 1
            spatial_size = (spatial_ratio * np.array([a, b, c])).astype(int)
            # print(spatial_size, images.shape, images.mean())
            # print(spatial_size)
            transform = Compose([
                RandAffined(
                    keys=['image', 'label'],
                    mode=('bilinear', 'nearest'),
                    spatial_size=spatial_size,
                    prob=1.,
                    translate_range=(0.1, 0.1, 0.1),
                    rotate_range=(np.pi / 20, np.pi / 20, np.pi / 20),
                    scale_range=.2,
                    # scale_range=[(2., 2.5), (2., 2.5), (2., 2.5)],
                    padding_mode='zeros',
                    allow_missing_keys=True,
                ),
                DivisiblePadd(
                    keys=['image', 'label'],
                    k=32,
                    mode='constant',
                    allow_missing_keys=True
                )
            ])
            if (a * b * c) >= (320 * 288 * 256):
                transform = self.transform
            else:
                transform = self.pad_transform
        else:
            transform = self.val_transform

        device = images.device

        with torch.no_grad():
            nb = images.shape[0]

            image_list = []
            mask_list = []
            cls_list = []
            bboxes_list = []
            bbox_weights_list = []
            batch_idx_list = []

            for i in range(nb):
                transformed_data = transform({
                    'image': images[i],
                    'label': masks[i],
                })

                image = transformed_data['image']
                mask = transformed_data['label'][0].as_tensor()

                orig_unique = batch['unique'][i]
                orig_count = batch['counts'][i]
                unique, counts = torch.unique(mask, sorted=True, return_counts=True)
                scale = torch.abs(torch.det(image.affine[:3, :3]))

                orig_count_dict = {k.item(): v.item() for k, v in zip(orig_unique, orig_count)}
                area_ratio = [c * scale / orig_count_dict[u.item()] for u, c in zip(unique, counts) if u.item() != 0]
                if len(area_ratio) == 0:
                    print(batch['path'])
                    area_ratio = torch.zeros(0, device=device).to(torch.float32)
                else:
                    area_ratio = torch.stack(area_ratio).clamp(0., 1.).to(torch.float32)

                # vital
                mask = torch.searchsorted(unique, mask)
                cls = unique[unique != 0]

                bboxes, mids = extract_bbox_from_mask(mask)
                image_list.append(image.as_tensor())
                mask_list.append(mask)
                bbox_weights_list.append(area_ratio)
                # print(area_ratio.cpu().numpy().tolist())
                # print(scale)
                # print(image.affine.cpu().numpy())

                center = bboxes.mean(axis=1)
                whd = bboxes[:, 1] - bboxes[:, 0]

                xyzwhd = torch.cat([center, whd], dim=1)
                img_size = torch.tensor(image.shape[1:], dtype=torch.float32, device=device)
                xyzwhd[:, :3] = center / img_size[None]
                xyzwhd[:, 3:6] = whd / img_size[None]
                bboxes_list.append(xyzwhd)

                # cls_list.append(torch.zeros(len(mids), dtype=torch.long, device=device))

                # tids
                cls_list.append((cls - 1).to(torch.long).to(device))
                batch_idx_list.append(torch.ones(len(mids), dtype=torch.long, device=device) * i)

                b = xyzwhd.clone()
                b[:, :3] = b[:, :3] * img_size[None]
                b[:, 3:6] = b[:, 3:6] * img_size[None]
                # if True:
                #     print(mode)
                #     show_3d(image, mask[None])
                #     vis_mask(mask.clone(), b, mids, mode='xyzwhd')
            del images, masks

            out_batch = {
                'images': torch.stack(image_list, dim=0),
                'masks': torch.stack(mask_list, dim=0),

                'bboxes': torch.cat(bboxes_list, dim=0),  # b, l, xyz, whd
                'weights': torch.cat(bbox_weights_list, dim=0),
                'cls': torch.cat(cls_list, dim=0),  # b, l, xyz, whd
                'batch_idx': torch.cat(batch_idx_list, dim=0),
                'path': batch['path'],
                'shape': batch["image"].shape,
            }
            # print(batch['path'], out_batch['images'].shape, out_batch['bboxes'].shape, device)
            return out_batch

    def get_loss(self, batch, mode='train'):
        if self.seg_loss is None:
            self.seg_loss = SegLoss(model=self.model)

        images, masks = batch["images"], batch["masks"]
        device = batch["images"].device
        pred = self.yolo.forward(images)

        loss_dict = self.seg_loss(pred, batch)

        # proto = F.interpolate(proto, scale_factor=2)
        # masks = F.interpolate(masks, scale_factor=0.5)
        # proto = F.interpolate(proto, scale_factor=4)

        loss_names = ['box', 'seg', 'cls', 'dfl']
        loss_weights = [4, 10, 4., 1.]
        # loss_weights = [4, 4, 1., 1.]

        loss = torch.zeros(1, device=device)
        for k, w in zip(loss_names, loss_weights):
            if k not in loss_dict:
                continue

            self.log(f'{mode}/{k}', loss_dict[k])
            loss += loss_dict[k] * w

        if mode == 'train':
            lr_list = [x['lr'] for x in self.optimizers().param_groups]
            for i, lr in enumerate(lr_list):
                self.log(f'lr/{i}', lr)
            self.log('loss', loss.item())
        else:
            self.log('val_loss', loss.item())

        loss_dict['loss'] = loss
        return loss_dict, pred

    def training_step(self, batch, batch_idx):
        batch = self.process_input_batch(batch, mode='train', show=False)
        loss, _ = self.get_loss(batch, mode='train')
        # print(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        batch = self.process_input_batch(batch, mode='val', show=False)
        loss, infer_out = self.get_loss(batch, mode='val')

        from monai.data.box_utils import non_max_suppression
        box = infer_out['box'][0].T
        score = infer_out['soft'][0,]
        score = torch.max(score, dim=0)[0]

        _, _, h, w, d = batch['images'].shape

        # neg_mask = (box[:, 3] < box[:, 0]) | (box[:, 4] < box[:, 1]) | (box[:, 5] < box[:, 2])
        # score[neg_mask] = 0

        score_mask = score > 0.3
        print('\n num candidate ', (score > .3).sum().item())

        box = box[score_mask]
        score = score[score_mask]

        res = non_max_suppression(box, score, nms_thresh=0.4)
        print('after nms', len(res), len(batch['bboxes']))

        # vis_mask(batch['masks'][0], batch['bboxes'], mode='xyzwhd')
        pred_bbox = box[res]
        # vis_mask(batch['masks'][0], pred_bbox, mode='xyzxyz')

        return loss


if __name__ == '__main__':
    from pytorch_lightning.callbacks import ModelCheckpoint, Callback
    from cbct.configs import *

    cfg = Config()
    cfg.LOAD_FROM = None
    cfg.NAME = 'tid'
    cfg.VERSION = 'fp16_fine'
    cfg.LR = 1e-3
    # cfg = TeethInst()

    print(socket.gethostname())
    cfg.SPATIAL_SIZE = (320, 320, 320)
    cfg.GPUS = [0, 1, 2, 3]
    cfg.BATCH_SIZE = 1
    cfg.DATA_ROOT = ''
    cfg.VERSION = 'stuck'


    cfg.EPOCHS = 100
    cfg.SHOW = True

    # net = PLModel.load_from_checkpoint(cfg.RESUME_FROM, cfg=cfg)
    if cfg.LOAD_FROM is None:
        net = PLModel(cfg)
    else:
        net = PLModel(cfg)
        pretrained_state_dict = torch.load(cfg.LOAD_FROM, map_location='cpu')['state_dict']

        import torch
        from ultralytics.yolo.utils.torch_utils import intersect_dicts # 智能合并权重字典，只合并相同形状的权重

        intersect_state_dict = {}
        for k, v in net.state_dict().items():
            if k not in pretrained_state_dict:
                print('missing key: {}'.format(k))
            elif v.shape != pretrained_state_dict[k].shape:
                print('mismatched shape: {} from {} to {}'.format(k, pretrained_state_dict[k].shape, v.shape))
            else:
                intersect_state_dict[k] = pretrained_state_dict[k]

        state_dict = intersect_dicts(intersect_state_dict, net.state_dict(), exclude=[])  # intersect
        net.load_state_dict(state_dict, strict=False)

    print(cfg.get_snapshot())
    # 创建TensorBoard日志记录器
    tb_logger = pytorch_lightning.loggers.TensorBoardLogger(
        save_dir=cfg.WEIGHT_DIR, # 日志保存路径
        name=cfg.NAME, # 实验名称
        flush_secs=1, # 每秒刷新日志
        version=cfg.VERSION # 版本号
    )
    # 模型检查点回调，保存最佳模型
    checkpoint_callback = ModelCheckpoint(
        dirpath=tb_logger.log_dir + '/checkpoints',
        filename='{epoch:03d}-{val_loss:.4f}-{val_dice:.4f}', # (epoch-验证损失-验证dice)
        save_top_k=4, # 保存前4个最佳模型
        monitor='val_loss', # 监控验证损失
        mode='min', # 最小化验证损失
        save_last=True, # 保存最后一个模型
    )

    # initialise Lightning's trainer.
    # 训练器配置
    trainer = pytorch_lightning.Trainer(
        # strategy='ddp_spawn',
        strategy="ddp_find_unused_parameters_false", # 分布式训练策略
        accelerator='gpu', # 加速器
        devices=1, # 使用1个GPU
        max_epochs=cfg.EPOCHS, # 最大训练轮数
        logger=tb_logger, # 日志记录器
        callbacks=[checkpoint_callback], # 回调函数列表
        sync_batchnorm=True, # 同步批归一化
        num_sanity_val_steps=2, # 训练前验证步数
        log_every_n_steps=10, # 每10步记录一次日志
        check_val_every_n_epoch=1, # 每1个epoch验证一次
        precision=16, # 使用混合精度训练（FP16 半精度浮点数）
    )

    # train
    # 开始训练
    trainer.fit(net,
                ckpt_path=cfg.LOAD_FROM
                )
    print(
        f"train completed, best_metric: {net.best_val_dice:.4f} "
        f"at epoch {net.best_val_epoch}")
