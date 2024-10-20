import math
from torch.optim.lr_scheduler import _LRScheduler


# 余弦退火学习率
class CosineAnnealingWarmupLR(_LRScheduler):
    def __init__(self, optimizer, total_iters, warmup_iters, eta_min=0, mode='linear', last_epoch=-1):
        self.mode = mode
        self.warmup_iter = warmup_iters
        self.eta_min = eta_min
        self.total_iters = total_iters

        super(CosineAnnealingWarmupLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.mode == 'exp':
            w = min(1, 1 - (math.exp(-(self.last_epoch) / self.warmup_iter)))
        elif self.mode == 'linear':
            w = min(1, self.last_epoch / self.warmup_iter)

        return [w * (self.eta_min + (base_lr - self.eta_min) *
                     (1 + math.cos(math.pi * self.last_epoch / self.total_iters)) / 2)
                for base_lr in self.base_lrs]
