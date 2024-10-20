from copy import copy
import torch
import torch.nn.functional as F
from ultralytics.yolo.utils.ops import crop_mask, xyxy2xywh, xyzwhd2xyzxyz, xyzxyz2xyzwhd, loose_mask
from ultralytics.yolo.v8.detect.train import Loss
import torch.nn as nn
from copy import copy
from cbct.utils.common import vis_mask
from cbct.task_aligned_focal_loss import TaskAlignedFocalLoss, task_aigned_focal_loss
import numpy as np
import torch
from ultralytics.yolo.utils.loss import BboxLoss
from ultralytics.yolo.utils.ops import xywh2xyxy
from ultralytics.yolo.utils.plotting import plot_images, plot_labels, plot_results
from ultralytics.yolo.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors
from ultralytics.yolo.utils.torch_utils import de_parallel, torch_distributed_zero_first
from monai.losses import DiceLoss, DiceFocalLoss, DiceCELoss, FocalLoss
from torch.utils.checkpoint import checkpoint


class SegLoss():

    def __init__(self, model, overlap=True):  # model must be de-paralleled
        super(SegLoss, self).__init__()
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        # BCEobj = nn.BCEWithLogitsLoss(reduction='none')
        # self.bce = FocalLoss(BCEobj, gamma=1.5, alpha=0.25)
        # self.bce = TaskAlignedFocalLoss()

        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.reg_max = m.reg_max
        self.reg_num = m.reg_num
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=20, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)
        self.max_inst_num = 50

        self.nm = model.model[-1].nm  # number of masks
        self.overlap = overlap
        self.dice_loss = DiceLoss(
            sigmoid=False,
            reduction='none',
        )

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 7, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 7, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:7] = xyzwhd2xyzxyz(out[..., 1:7].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            proj = self.proj.type(pred_dist.dtype).to(pred_dist.device)
            pred_dist = pred_dist.view(b, a, 6, c // 6).softmax(3).matmul(proj)
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl
        dfl = preds['dfl']
        cls = preds['cls']
        proto = preds['proto']
        pred_masks = preds['mc']

        # batch_size, _, mask_h, mask_w, mask_d = proto.shape  # batch size, number of masks, mask height, mask width
        batch_size = proto.size(0)
        pred_distri = torch.cat([xi.view(dfl[0].shape[0], self.reg_max * self.reg_num, -1) for xi in dfl], 2)
        pred_scores = torch.cat([xi.view(cls[0].shape[0], self.nc, -1) for xi in cls], 2)
        # b, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(dfl[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(dfl, self.stride, 0.5)

        # targets
        batch_idx = batch['batch_idx'].view(-1, 1)
        targets = torch.cat((batch_idx, batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[0, 1, 2, 0, 1, 2]])
        gt_labels, gt_bboxes = targets.split((1, 6), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        fg_mask = fg_mask.bool()
        target_scores_sum = max(target_scores.sum(), 1)
        masks = batch['masks'].to(self.device).float()

        # values, indices= torch.topk(torch.sigmoid(pred_scores[fg_mask])[:, 0], 10)
        # comp = target_scores[fg_mask][indices][:, 0]
        # print('score', fg_mask.sum().item())
        # for a, b in zip(values, comp):
        #     print(f'{a.item():.4f} {b.item():.4f}', end=' | ')
        # print()
        #
        # values, indices= torch.topk(target_scores[fg_mask][:, 0], 10)
        # comp = torch.sigmoid(pred_scores[fg_mask])[indices][:, 0]
        # print('rev')
        # for a, b in zip(values, comp):
        #     print(f'{a.item():.4f} {b.item():.4f}', end=' | ')
        # print()

        # print('\n fg', fg_mask.sum().item(), target_scores_sum.item(), len(torch.unique(target_gt_idx)))
        # from cbct.utils.common import vis_mask
        # vis_mask(masks[0], bboxes=batch['bboxes'] * imgsz[[2, 1, 0, 2, 1, 0]], mids=batch['cls'] + 1, mode = 'xyzwhd')
        # vis_mask(masks[0], bboxes=gt_bboxes[0], mids=gt_labels[0] + 1, mode='xyzxyz')
        # vis_mask(masks[0], bboxes=target_bboxes[fg_mask], mids=target_gt_idx[fg_mask] + 1, mode='xyzxyz')

        # soft = torch.sigmoid(pred_scores[fg_mask])
        # s, l = torch.max(soft, dim=-1)
        # vis_mask(masks[0], bboxes=(pred_bboxes * stride_tensor)[fg_mask].detach() , mids=target_gt_idx[fg_mask].detach() + 1, mode='xyzxyz')


        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way

        # cls_loss = self.bce(pred_scores, target_scores.to(dtype))
        weights = torch.zeros_like(target_scores)

        if len(batch['weights']) != 0:
            weights[fg_mask] = batch['weights'][target_gt_idx[fg_mask]][..., None]
        else:
            print(batch['path'])

        target_scores = target_scores * weights
        # cls_loss = task_aigned_focal_loss(pred_scores, target_scores ).mean()

        # loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE
        pos_weight = torch.tensor(10, device=self.device, dtype=dtype)
        cls_loss = F.binary_cross_entropy_with_logits(pred_scores, target_scores.to(dtype),
                                           pos_weight=pos_weight,
                                           reduction='none').sum() / target_scores_sum / 10 # BCE
        loss[2] = cls_loss
        # print(loss[2].item())

        if fg_mask.sum():
            # bbox loss
            loss[0], loss[3] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes / stride_tensor,
                                              target_scores, target_scores_sum, fg_mask)
            # masks loss
            # masks = F.interpolate(masks[None], scale_factor=.5, mode='nearest')[0]
            # proto = F.interpolate(proto, scale_factor=2)

            # downsample if needed for gpu memory
            # _, _, mask_h, mask_w, mask_d = proto.shape  # batch size, number of masks, mask height, mask width
            # if tuple(masks.shape[-3:]) != (mask_h, mask_w, mask_d):
            #     proto = F.interpolate(proto, masks.shape[-3:], mode='trilinear', align_corners=False)
            #     masks = F.interpolate(masks[None], (mask_h, mask_w, mask_d), mode='nearest')[0]

            for i in range(batch_size):
                fg_sum = fg_mask[i].sum()
                print(fg_sum, batch['path'], batch['images'].shape, batch['shape'], len(batch['bboxes']), ' ')

                if fg_sum > 5:
                    idx = torch.where(fg_mask[i])[0]
                    p = torch.randperm(len(idx))
                    shuffled_idx = idx[p]

                    seg_loss = torch.tensor(0., dtype=loss.dtype, device=self.device)
                    seg_count = torch.zeros_like(seg_loss)

                    # for j in range(0, len(shuffled_idx), step):
                    sample = shuffled_idx[:600] # change it for your gpus

                    mask_idx = target_gt_idx[i][sample]
                    mweight = weights[i][sample]
                    p_masks = pred_masks[i][sample]
                    mask_i = masks[[i]]
                    proto_i = proto[i]

                    xyzxyzn = target_bboxes[i][sample] / imgsz[[0, 1, 2, 0, 1, 2]]
                    _, mask_h, mask_w, mask_d = masks.shape  # batch size, number of masks, mask height, mask width
                    mxyxy = xyzxyzn * torch.tensor([mask_h, mask_w, mask_d, mask_h, mask_w, mask_d],
                                                   device=self.device)

                    num_inst = mask_idx.shape[0]
                    batch_loss = self.seg_forward(mask_i, mask_idx, mweight, p_masks, proto_i, mxyxy,
                                                  )

                    seg_loss += batch_loss * num_inst
                    seg_count += num_inst

                    seg_loss = seg_loss / seg_count
                    loss[1] += seg_loss
                    # loss[1] += self.single_mask_loss(gt_mask, pred_masks[i][sample], proto[i], mxyxy, marea)# seg

                # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
                else:
                    loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss
        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:

            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss
            loss[0] += (pred_distri * 0).sum()

        return {
            'box': loss[0],
            'seg': loss[1] / batch_size,
            'cls': loss[2],
            'dfl': loss[3],
        }

    def seg_forward(self, mask_i, mask_idx, mweight, p_masks, proto_i, mxyxy):
        gt_mask = torch.where(mask_i == (mask_idx + 1).view(-1, 1, 1, 1), 1.0, 0.0)
        batch_loss = self.single_mask_loss(gt_mask, p_masks, proto_i, mxyxy, mweight)
        return batch_loss

    def single_mask_loss(self, gt_mask, pred, proto, xyxy, mweight):
        # Mask loss for one image
        mweight = mweight[:, 0]
        mweight_sum = mweight.sum() + 1e-7

        pred_mask = (pred @ proto.view(self.nm, -1)).view(-1, *proto.shape[1:])  # (n,32) @ (32,80,80) -> (n,80,80)
        pred_mask = F.interpolate(pred_mask[:, None], scale_factor=2, mode='trilinear')[:, 0]
        dilate_mask = loose_mask(pred_mask, xyxy, loose_factor=0.5)

        bce_loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        crop_bce_loss = ((bce_loss * dilate_mask).sum(dim=(1,2,3)) / dilate_mask.sum(dim=(1,2,3)))
        crop_bce_loss = (crop_bce_loss * mweight).sum() / mweight_sum

        dice_loss = self.dice_loss((pred_mask[:, None].sigmoid() * dilate_mask[:, None]), gt_mask[:, None] * dilate_mask[:, None])
        dice_loss = (dice_loss.view(-1) * mweight).sum() / mweight_sum
        # print(dice_loss.item(), crop_bce_loss.item())
        # print(crop_bce_loss.item(), dice_loss.item())

        loss = crop_bce_loss * 2 + dice_loss
        return loss
        # return crop_bce_loss + dice_loss


    def single_mask_rel_coord_loss(self, gt_mask, pred,  proto, xyxy, anchors, prior, area):
        # Mask loss for one image
        pred_mask = (pred @ proto.view(self.nm, -1)).view(-1, *proto.shape[1:])  # (n,32) @ (32,80,80) -> (n,80,80)

        _, _, h, w, d = feats[i].shape
        sd = torch.arange(end=d, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sw = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sh = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sh, sw, sd = torch.meshgrid(sh, sw, sd, indexing='ij')

        dilate_mask = loose_mask(pred_mask, xyxy, loose_factor=0.5)

        bce_loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        crop_bce_loss = ((bce_loss * dilate_mask).sum(dim=(1,2,3)) / dilate_mask.sum(dim=(1,2,3))).mean()

        dice_loss = self.dice_loss(pred_mask.sigmoid(), gt_mask)
        print(bce_loss.item(), dice_loss.item())
        # print(dice_loss.item(), crop_bce_loss.item())

        return crop_bce_loss + dice_loss