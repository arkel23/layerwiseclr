import argparse
import pytorch_lightning as pl

def ret_args(ret_parser=False):

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=0, help='random seed for initialization')
    
    parser.add_argument('--dataset_name', choices=['cifar10', 'cifar100', 'imagenet'], 
                        default='cifar10', help='Which dataset to use.')
    parser.add_argument('--dataset_path', help='Path for the dataset.')
    
    parser.add_argument('--model_name', choices=['B_16', 'B_32', 'L_16', 'L_32'], default='B_16',
                        help='Which model architecture to use')
    
    parser.add_argument('--image_size', default=128, type=int,
                        help='Image (square) resolution size')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size for train/val/test.')
    
    parser.add_argument('--optimizer', choices=['sgd', 'adam'], default='sgd')
    parser.add_argument('--learning_rate', default=0.001, type=float,
                        help='Initial learning rate.')  
    parser.add_argument('--weight_decay', type=float, default=0.00)
    parser.add_argument('--warmup_steps', type=int, default=1000, help='Warmup steps for LR scheduler.')
    
    parser.add_argument('--pretrained',action='store_true',
                        help='Loads pretrained model if available')
    parser.add_argument('--checkpoint_path', type=str, default=None)     
    parser.add_argument('--transfer_learning', action='store_true',
                        help='Load partial state dict for transfer learning'
                        'Resets the [embeddings, logits and] fc layer for ViT')    
    parser.add_argument('--load_partial_mode', choices=['full_tokenizer', 'patchprojection', 
                        'posembeddings', 'clstoken', 'patchandposembeddings', 
                        'patchandclstoken', 'posembeddingsandclstoken', None], default=None,
                        help='Load pre-processing components to speed up training')
    parser.add_argument("--deit_recipe", action='store_true',
                        help="Use DeiT training recipe"
    
    parser.add_argument('--results_dir', default='results_training', type=str,
                        help='The directory where results will be stored')
    parser.add_argument('--save_checkpoint_freq', default=100, type=int,
                        help='Frequency (in epochs) to save checkpoints')

    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(gpus=1, max_epochs=2, gradient_clip_val=1.0)
    parser.set_defaults(precision=16, log_gpu_memory=None, profiler=None)

    if ret_parser:
        return parser
    args = parser.parse_args()

    args.run_name = '{}_{}_is{}_bs{}_{}lr{}_pt{}_seed{}'.format(
    args.dataset_name, args.model_name, args.image_size, args.batch_size, 
    args.optimizer, args.learning_rate, args.pretrained, args.seed)

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

