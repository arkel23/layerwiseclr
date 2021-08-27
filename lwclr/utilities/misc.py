import os
import argparse

import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

import lwclr.models as models
from .datamodules import CIFAR10DM, CIFAR100DM, ImageNetDM

def ret_args(ret_parser=False):

    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, 
                        choices=['simclr', 'simlwclr', 'lwplclr', 'linear_eval', 'fine_tuning'],
                        default='simlwclr', help='Framework for training and evaluation')

    parser.add_argument('--seed', type=int, default=0, help='random seed for initialization')
    parser.add_argument('--no_cpu_workers', type=int, default=4, help='CPU workers for data loading.')

    parser.add_argument('--results_dir', default='results_training', type=str,
                        help='The directory where results will be stored')
    parser.add_argument('--save_checkpoint_freq', default=100, type=int,
                        help='Frequency (in epochs) to save checkpoints')

    parser.add_argument('--dataset_name', choices=['cifar10', 'cifar100', 'imagenet'], 
                        default='cifar10', help='Which dataset to use.')
    parser.add_argument('--dataset_path', help='Path for the dataset.')
    parser.add_argument("--deit_recipe", action='store_true',
                        help="Use DeiT training recipe")
    
    parser.add_argument('--image_size', default=224, type=int,
                        help='Image (square) resolution size')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size for train/val/test.')
    
    parser = models.LitSimLWCLR.add_model_specific_args(parser)
    
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(gpus=1, max_epochs=2, gradient_clip_val=1.0)
    parser.set_defaults(precision=32, log_gpu_memory=None, profiler=None)

    if ret_parser:
        return parser
    args = parser.parse_args()

    args.run_name = '{}_{}_{}_is{}_bs{}_{}lr{}_seed{}'.format(
        args.mode, args.dataset_name, args.model_name, args.image_size, args.batch_size, 
        args.optimizer, args.learning_rate, args.seed)

    if args.deit_recipe:
        ''' taken from DeiT paper
        https://arxiv.org/abs/2012.12877
        https://github.com/facebookresearch/deit/blob/main/main.py'''
        # augmentation and random erase params
        args.color_jitter = 0.4
        args.aa = 'rand-m9-mstd0.5-inc1'
        args.smoothing = 0.1
        args.train_interpolation = 'bicubic'
        args.repeated_aug = True
        args.reprob = 0.25
        args.remode = 'pixel'
        args.recount = 1
        args.resplit = False
        # mixup params
        args.mixup = 0.8
        args.cutmix = 1.0
        args.cutmix_minmax = None
        args.mixup_prob = 1.0
        args.mixup_switch_prob = 0.5
        args.mixup_mode = 'batch'

    return args

def load_trainer(args, model, wandb_logger):
    # https://lightning-bolts.readthedocs.io/en/latest/self_supervised_callbacks.html
    # https://github.com/PyTorchLightning/lightning-bolts/blob/47eb2aae677350159c9ec0dc8ccdb6eef4217fff/pl_bolts/callbacks/ssl_online.py#L66
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.results_dir, args.run_name), 
        filename='{epoch}', monitor='val_loss', save_on_train_epoch_end=False, 
        verbose=True, save_top_k=-1, save_last=True, save_weights_only=True,
        mode='min', every_n_epochs=args.save_checkpoint_freq)

    lr_monitor = LearningRateMonitor(logging_interval='step')

    if args.mode not in ['linear_eval', 'fine_tuning']:
        online_eval_callback = models.SSLOnlineLinearEvaluator(
            mode=args.mode, z_dim=model.backbone.configuration.hidden_size, 
            num_classes=args.num_classes, lr=args.learning_rate)
        trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback, lr_monitor, online_eval_callback])
    else:
        trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback, lr_monitor])
    trainer.logger = wandb_logger
    
    return trainer


def load_plmodel(args):
    if args.mode == 'simlwclr':
        model = models.LitSimLWCLR(args)
    elif args.mode == 'simclr':
        model = models.LitSimCLR(args)
    elif args.mode == 'lwplclr':
        model = models.LitLWPLCLR(args)
    elif args.mode == 'linear_eval' or args.mode == 'fine_tuning':
        model = models.LitEvaluator(args)
    return model


def return_prepared_dm(args):

    assert args.dataset_path, "Dataset path must not be empty."
    # setup data
    if args.dataset_name == 'cifar10':
        dm = CIFAR10DM(args)
    elif args.dataset_name == 'cifar100':
        dm = CIFAR100DM(args)
    elif args.dataset_name == 'imagenet':
        dm = ImageNetDM(args)
    
    dm.prepare_data()
    dm.setup('fit')
    args.num_classes = dm.num_classes

    if args.max_steps:
        total_steps = args.max_steps
    else:
        total_steps = args.max_epochs * len(dm.train_dataloader())
    args.total_steps = total_steps

    if args.warmup_epochs:
        args.warmup_steps = args.max_epochs * len(dm.train_dataloader())
    
    return dm


def environment_loader(args, init=True):
        
    # set up W&B logger
    if init:
        os.makedirs(args.results_dir, exist_ok=True)
        wandb.init(config=args)
        wandb.run.name = args.run_name
        wandb_logger = WandbLogger(name=args.run_name)
        
    # seed everything
    pl.seed_everything(seed=args.seed)

    # prepare datamodule
    dm = return_prepared_dm(args)

    # setup model and trainer
    model = load_plmodel(args)
    trainer = load_trainer(args, model, wandb_logger)
    
    return dm, trainer, model
