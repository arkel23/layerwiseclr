import math

from torch.optim.lr_scheduler import LambdaLR
from timm.optim import create_optimizer_v2

class ConstantLRSchedule(LambdaLR):
    """ Constant learning rate schedule.
    """
    def __init__(self, optimizer, last_epoch=-1):
        super(ConstantLRSchedule, self).__init__(optimizer, lambda _: 1.0, last_epoch=last_epoch)

class ConstantEpochDecayLRSchedule(LambdaLR):
    """ Linear warmup and then constant.
        Linearly increases learning rate schedule from 0 to 1 over `warmup_steps` training steps.
        Keeps learning rate schedule equal to 1. after warmup_steps.
    """
    def __init__(self, optimizer, steps_per_epoch, epochs=[150, 180, 120], last_epoch=-1):
        self.decay_epochs = epochs
        self.decay_steps = [epoch * steps_per_epoch for epoch in self.decay_epochs]
        super(ConstantEpochDecayLRSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.decay_steps[0]:
            return 1
        elif step > self.decay_steps[0] and step < self.decay_steps[1]:
            return 0.1
        elif step > self.decay_steps[1] and step < self.decay_steps[2]:
            return 0.01
        else:
            return 0.001 

class WarmupConstantSchedule(LambdaLR):
    """ Linear warmup and then constant.
        Linearly increases learning rate schedule from 0 to 1 over `warmup_steps` training steps.
        Keeps learning rate schedule equal to 1. after warmup_steps.
    """
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(WarmupConstantSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        return 1.

class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))

def create_scheduler(args, optimizer, steps_total):
    if args.lr_scheduler == "warmup_cosine":
        lr_scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=steps_total)
    elif args.lr_scheduler == 'warmup_constant':
        lr_scheduler = WarmupConstantSchedule(optimizer, warmup_steps=args.warmup_steps)
    elif args.lr_scheduler == 'constant_decay':
        lr_scheduler = ConstantEpochDecayLRSchedule(optimizer, steps_per_epoch=steps_total//args.max_epochs)
    else:
        lr_scheduler = ConstantLRSchedule(optimizer)
    scheduler = {'scheduler': lr_scheduler, 
                 'name': 'learning_rate', 'interval': 'step', 'frequency': 1}
    return scheduler

def create_optim(model, args):
    optimizer = create_optimizer_v2(model, args.optimizer, 
                    args.learning_rate, weight_decay=args.weight_decay, filter_bias_and_bn=True)
    return optimizer