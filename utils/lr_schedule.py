import math
from torch.optim.lr_scheduler import _LRScheduler

class WarmupCosineScheduler(_LRScheduler):
    """
    Warmup + Cosine Annealing LR scheduler.
    
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_epochs (int): Number of epochs for linear warmup.
        total_epochs (int): Total number of training epochs.
        min_lr (float): Minimum learning rate at the end of cosine annealing. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=0.0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        current_epoch = self.last_epoch

        if current_epoch < self.warmup_epochs:
            # Linear warmup: lr = base_lr * (epoch / warmup_epochs)
            alpha = current_epoch / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine annealing: from base_lr to min_lr
            progress = (current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return [self.min_lr + (base_lr - self.min_lr) * cosine_decay for base_lr in self.base_lrs]


