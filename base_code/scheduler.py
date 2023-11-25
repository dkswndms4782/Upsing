from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR
import torch.optim.lr_scheduler
#  from transformers import get_linear_schedule_with_warmup

def get_scheduler(optimizer, args):
    if args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.5, mode='max', verbose=True)
 #    elif args.scheduler == 'linear_warmup':
 #        scheduler = get_linear_schedule_with_warmup(optimizer,
 #                                                   num_warmup_steps=args.warmup_steps,
 #                                                   num_training_steps=args.total_steps)
    elif args.scheduler == 'cyclic_lr':
        scheduler = CyclicLR(optimizer, base_lr=args.lr, max_lr=0.1, step_size_up=50, step_size_down=100, mode='triangular')
    return scheduler