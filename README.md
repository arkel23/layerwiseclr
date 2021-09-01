# LayerWiseCLR
ICLR CSS 2022 Project on Layer-Wise Contrastive Learning of Representations for Vision Transformers

## Setup:
Run the `setup.sh` file to install the PyTorch-Pretrained-ViT and others from the requirements.txt file. 
Make sure pytorch, torchvision, numpy, are all installed.

## Training
All of these use ViT B-16 as default, to change the `--model_name MODEL` argument.
Train a SimCLR model on CIFAR10, image size=128, batch size=128, for 300 epochs and saving each 50 epochs:
```
python train.py --gpus 1 --image_size 128 --dataset_path data --max_epochs 300 --dataset_name cifar10 --mode simclr --batch_size 128 --save_checkpoint_freq 50
```

Train a LWCLR twin model with full supervision with batch size=128, with contrast between the last layer of both models and all previous settings:
```
python train.py --gpus 1 --image_size 128 --dataset_path data --max_epochs 300 --dataset_name cifar10 --mode lwclr_full_single --batch_size 128 --save_checkpoint_freq 50
```

Train a LWCLR twin model with SimCLR contrastive supervision for auxiliary model with batch size=64, with contrast between the last layer of both models and all previous settings:
```
python train.py --gpus 1 --image_size 128 --dataset_path data --max_epochs 300 --dataset_name cifar10 --mode lwclr_cont_single --batch_size 64 --save_checkpoint_freq 50
```

## Description of models
### SimCLR
Takes two augmentations from same image for positive pairs, and other images in batch for negative pairs.

### LWCLR
Trains two models, each with their independent weights, but same architecture, one called auxiliary or generator, and the other called discriminator. Auxiliary is trained either using full supervision or contrastive similar to SimCLR, and it provides representations from either the last layer or from any intermediate layer, to contrast against the discriminator, which learns through a contrastive loss from its feature map from the last layer of either the same image, or a different augmentation of the same image.

## Full list of arguments
```
usage: train.py [-h] [--mode {simclr,lwclr_full_all,lwclr_full_single,lwclr_cont_all,lwclr_cont_single,linear_eval,fine_tuning}]
                [--seed SEED] [--no_cpu_workers NO_CPU_WORKERS] [--results_dir RESULTS_DIR]
                [--save_checkpoint_freq SAVE_CHECKPOINT_FREQ] [--dataset_name {cifar10,cifar100,imagenet}]
                [--dataset_path DATASET_PATH] [--deit_recipe] [--image_size IMAGE_SIZE] [--batch_size BATCH_SIZE]
                [--temperature TEMPERATURE] [--optimizer {sgd,adam}] [--learning_rate LEARNING_RATE] [--weight_decay WEIGHT_DECAY]
                [--warmup_steps WARMUP_STEPS] [--warmup_epochs WARMUP_EPOCHS]
                [--model_name {B_16,B_32,L_16,L_32,effnet_b0,resnet18,resnet50}] [--vit_avg_pooling] [--no_proj_layers {1,2,3}]
                [--layer_contrast LAYER_CONTRAST] [--random_layer_contrast] [--fs_weight FS_WEIGHT] [--pl_weight PL_WEIGHT]
                [--cont_weight CONT_WEIGHT] [--pretrained_checkpoint] [--checkpoint_path CHECKPOINT_PATH] [--transfer_learning]
                [--load_partial_mode {full_tokenizer,patchprojection,posembeddings,clstoken,patchandposembeddings,patchandclstoken,posembeddingsandclstoken,None}]
                [--interm_features_fc] [--conv_patching] [--logger [LOGGER]] [--checkpoint_callback [CHECKPOINT_CALLBACK]]
                [--default_root_dir DEFAULT_ROOT_DIR] [--gradient_clip_val GRADIENT_CLIP_VAL]
                [--gradient_clip_algorithm GRADIENT_CLIP_ALGORITHM] [--process_position PROCESS_POSITION] [--num_nodes NUM_NODES]
                [--num_processes NUM_PROCESSES] [--devices DEVICES] [--gpus GPUS] [--auto_select_gpus [AUTO_SELECT_GPUS]]
                [--tpu_cores TPU_CORES] [--ipus IPUS] [--log_gpu_memory LOG_GPU_MEMORY]
                [--progress_bar_refresh_rate PROGRESS_BAR_REFRESH_RATE] [--overfit_batches OVERFIT_BATCHES]
                [--track_grad_norm TRACK_GRAD_NORM] [--check_val_every_n_epoch CHECK_VAL_EVERY_N_EPOCH] [--fast_dev_run [FAST_DEV_RUN]]
                [--accumulate_grad_batches ACCUMULATE_GRAD_BATCHES] [--max_epochs MAX_EPOCHS] [--min_epochs MIN_EPOCHS]
                [--max_steps MAX_STEPS] [--min_steps MIN_STEPS] [--max_time MAX_TIME] [--limit_train_batches LIMIT_TRAIN_BATCHES]
                [--limit_val_batches LIMIT_VAL_BATCHES] [--limit_test_batches LIMIT_TEST_BATCHES]
                [--limit_predict_batches LIMIT_PREDICT_BATCHES] [--val_check_interval VAL_CHECK_INTERVAL]
                [--flush_logs_every_n_steps FLUSH_LOGS_EVERY_N_STEPS] [--log_every_n_steps LOG_EVERY_N_STEPS]
                [--accelerator ACCELERATOR] [--sync_batchnorm [SYNC_BATCHNORM]] [--precision PRECISION]
                [--weights_summary WEIGHTS_SUMMARY] [--weights_save_path WEIGHTS_SAVE_PATH]
                [--num_sanity_val_steps NUM_SANITY_VAL_STEPS] [--truncated_bptt_steps TRUNCATED_BPTT_STEPS]
                [--resume_from_checkpoint RESUME_FROM_CHECKPOINT] [--profiler PROFILER] [--benchmark [BENCHMARK]]
                [--deterministic [DETERMINISTIC]] [--reload_dataloaders_every_n_epochs RELOAD_DATALOADERS_EVERY_N_EPOCHS]
                [--reload_dataloaders_every_epoch [RELOAD_DATALOADERS_EVERY_EPOCH]] [--auto_lr_find [AUTO_LR_FIND]]
                [--replace_sampler_ddp [REPLACE_SAMPLER_DDP]] [--terminate_on_nan [TERMINATE_ON_NAN]]
                [--auto_scale_batch_size [AUTO_SCALE_BATCH_SIZE]] [--prepare_data_per_node [PREPARE_DATA_PER_NODE]] [--plugins PLUGINS]
                [--amp_backend AMP_BACKEND] [--amp_level AMP_LEVEL] [--distributed_backend DISTRIBUTED_BACKEND]
                [--move_metrics_to_cpu [MOVE_METRICS_TO_CPU]] [--multiple_trainloader_mode MULTIPLE_TRAINLOADER_MODE]
                [--stochastic_weight_avg [STOCHASTIC_WEIGHT_AVG]]

optional arguments:
  -h, --help            show this help message and exit
  --mode {simclr,lwclr_full_all,lwclr_full_single,lwclr_cont_all,lwclr_cont_single,linear_eval,fine_tuning}
                        Framework for training and evaluation
  --seed SEED           random seed for initialization
  --no_cpu_workers NO_CPU_WORKERS
                        CPU workers for data loading.
  --results_dir RESULTS_DIR
                        The directory where results will be stored
  --save_checkpoint_freq SAVE_CHECKPOINT_FREQ
                        Frequency (in epochs) to save checkpoints
  --dataset_name {cifar10,cifar100,imagenet}
                        Which dataset to use.
  --dataset_path DATASET_PATH
                        Path for the dataset.
  --deit_recipe         Use DeiT training recipe
  --image_size IMAGE_SIZE
                        Image (square) resolution size
  --batch_size BATCH_SIZE
                        Batch size for train/val/test.
  --temperature TEMPERATURE
                        temperature parameter for ntxent loss
  --optimizer {sgd,adam}
  --learning_rate LEARNING_RATE
                        Initial learning rate.
  --weight_decay WEIGHT_DECAY
  --warmup_steps WARMUP_STEPS
                        Warmup steps for LR scheduler.
  --warmup_epochs WARMUP_EPOCHS
                        If doing warmup in terms of epochs instead of steps.
  --model_name {B_16,B_32,L_16,L_32,effnet_b0,resnet18,resnet50}
                        Which model architecture to use
  --vit_avg_pooling     If use this flag then uses average pooling instead of cls token of ViT
  --no_proj_layers {1,2,3}
                        Number of layers for projection head.
  --layer_contrast LAYER_CONTRAST
                        Layer features for pairs
  --random_layer_contrast
                        If use this flag then at each step chooses a random layer from gen to contrast against
  --fs_weight FS_WEIGHT
                        Weight for fully supervised loss
  --pl_weight PL_WEIGHT
                        Wegith for layer-wise pseudolabels loss
  --cont_weight CONT_WEIGHT
                        Weight for contrastive loss
  --pretrained_checkpoint
                        Loads pretrained model if available
  --checkpoint_path CHECKPOINT_PATH
  --transfer_learning   Load partial state dict for transfer learningResets the [embeddings, logits and] fc layer for ViT
  --load_partial_mode {full_tokenizer,patchprojection,posembeddings,clstoken,patchandposembeddings,patchandclstoken,posembeddingsandclstoken,None}
                        Load pre-processing components to speed up training
  --interm_features_fc  If use this flag creates FC using intermediate features instead of only last layer.
  --conv_patching       If use this flag uses a small convolutional stem instead of single large-stride convolution for patch
                        projection.

pl.Trainer:
  --logger [LOGGER]     Logger (or iterable collection of loggers) for experiment tracking. A ``True`` value uses the default
                        ``TensorBoardLogger``. ``False`` will disable logging. If multiple loggers are provided and the `save_dir`
                        property of that logger is not set, local files (checkpoints, profiler traces, etc.) are saved in
                        ``default_root_dir`` rather than in the ``log_dir`` of any of the individual loggers.
  --checkpoint_callback [CHECKPOINT_CALLBACK]
                        If ``True``, enable checkpointing. It will configure a default ModelCheckpoint callback if there is no user-
                        defined ModelCheckpoint in :paramref:`~pytorch_lightning.trainer.trainer.Trainer.callbacks`.
  --default_root_dir DEFAULT_ROOT_DIR
                        Default path for logs and weights when no logger/ckpt_callback passed. Default: ``os.getcwd()``. Can be remote
                        file paths such as `s3://mybucket/path` or 'hdfs://path/'
  --gradient_clip_val GRADIENT_CLIP_VAL
                        0 means don't clip.
  --gradient_clip_algorithm GRADIENT_CLIP_ALGORITHM
                        'value' means clip_by_value, 'norm' means clip_by_norm. Default: 'norm'
  --process_position PROCESS_POSITION
                        orders the progress bar when running multiple models on same machine.
  --num_nodes NUM_NODES
                        number of GPU nodes for distributed training.
  --num_processes NUM_PROCESSES
                        number of processes for distributed training with distributed_backend="ddp_cpu"
  --devices DEVICES     Will be mapped to either `gpus`, `tpu_cores`, `num_processes` or `ipus`, based on the accelerator type.
  --gpus GPUS           number of gpus to train on (int) or which GPUs to train on (list or str) applied per node
  --auto_select_gpus [AUTO_SELECT_GPUS]
                        If enabled and `gpus` is an integer, pick available gpus automatically. This is especially useful when GPUs are
                        configured to be in "exclusive mode", such that only one process at a time can access them.
  --tpu_cores TPU_CORES
                        How many TPU cores to train on (1 or 8) / Single TPU to train on [1]
  --ipus IPUS           How many IPUs to train on.
  --log_gpu_memory LOG_GPU_MEMORY
                        None, 'min_max', 'all'. Might slow performance
  --progress_bar_refresh_rate PROGRESS_BAR_REFRESH_RATE
                        How often to refresh progress bar (in steps). Value ``0`` disables progress bar. Ignored when a custom progress
                        bar is passed to :paramref:`~Trainer.callbacks`. Default: None, means a suitable value will be chosen based on
                        the environment (terminal, Google COLAB, etc.).
  --overfit_batches OVERFIT_BATCHES
                        Overfit a fraction of training data (float) or a set number of batches (int).
  --track_grad_norm TRACK_GRAD_NORM
                        -1 no tracking. Otherwise tracks that p-norm. May be set to 'inf' infinity-norm.
  --check_val_every_n_epoch CHECK_VAL_EVERY_N_EPOCH
                        Check val every n train epochs.
  --fast_dev_run [FAST_DEV_RUN]
                        runs n if set to ``n`` (int) else 1 if set to ``True`` batch(es) of train, val and test to find any bugs (ie: a
                        sort of unit test).
  --accumulate_grad_batches ACCUMULATE_GRAD_BATCHES
                        Accumulates grads every k batches or as set up in the dict.
  --max_epochs MAX_EPOCHS
                        Stop training once this number of epochs is reached. Disabled by default (None). If both max_epochs and
                        max_steps are not specified, defaults to ``max_epochs`` = 1000.
  --min_epochs MIN_EPOCHS
                        Force training for at least these many epochs. Disabled by default (None). If both min_epochs and min_steps are
                        not specified, defaults to ``min_epochs`` = 1.
  --max_steps MAX_STEPS
                        Stop training after this number of steps. Disabled by default (None).
  --min_steps MIN_STEPS
                        Force training for at least these number of steps. Disabled by default (None).
  --max_time MAX_TIME   Stop training after this amount of time has passed. Disabled by default (None). The time duration can be
                        specified in the format DD:HH:MM:SS (days, hours, minutes seconds), as a :class:`datetime.timedelta`, or a
                        dictionary with keys that will be passed to :class:`datetime.timedelta`.
  --limit_train_batches LIMIT_TRAIN_BATCHES
                        How much of training dataset to check (float = fraction, int = num_batches)
  --limit_val_batches LIMIT_VAL_BATCHES
                        How much of validation dataset to check (float = fraction, int = num_batches)
  --limit_test_batches LIMIT_TEST_BATCHES
                        How much of test dataset to check (float = fraction, int = num_batches)
  --limit_predict_batches LIMIT_PREDICT_BATCHES
                        How much of prediction dataset to check (float = fraction, int = num_batches)
  --val_check_interval VAL_CHECK_INTERVAL
                        How often to check the validation set. Use float to check within a training epoch, use int to check every n
                        steps (batches).
  --flush_logs_every_n_steps FLUSH_LOGS_EVERY_N_STEPS
                        How often to flush logs to disk (defaults to every 100 steps).
  --log_every_n_steps LOG_EVERY_N_STEPS
                        How often to log within steps (defaults to every 50 steps).
  --accelerator ACCELERATOR
                        Previously known as distributed_backend (dp, ddp, ddp2, etc...). Can also take in an accelerator object for
                        custom hardware.
  --sync_batchnorm [SYNC_BATCHNORM]
                        Synchronize batch norm layers between process groups/whole world.
  --precision PRECISION
                        Double precision (64), full precision (32) or half precision (16). Can be used on CPU, GPU or TPUs.
  --weights_summary WEIGHTS_SUMMARY
                        Prints a summary of the weights when training begins.
  --weights_save_path WEIGHTS_SAVE_PATH
                        Where to save weights if specified. Will override default_root_dir for checkpoints only. Use this if for
                        whatever reason you need the checkpoints stored in a different place than the logs written in
                        `default_root_dir`. Can be remote file paths such as `s3://mybucket/path` or 'hdfs://path/' Defaults to
                        `default_root_dir`.
  --num_sanity_val_steps NUM_SANITY_VAL_STEPS
                        Sanity check runs n validation batches before starting the training routine. Set it to `-1` to run all batches
                        in all validation dataloaders.
  --truncated_bptt_steps TRUNCATED_BPTT_STEPS
                        Deprecated in v1.3 to be removed in 1.5. Please use
                        :paramref:`~pytorch_lightning.core.lightning.LightningModule.truncated_bptt_steps` instead.
  --resume_from_checkpoint RESUME_FROM_CHECKPOINT
                        Path/URL of the checkpoint from which training is resumed. If there is no checkpoint file at the path, start
                        from scratch. If resuming from mid-epoch checkpoint, training will start from the beginning of the next epoch.
  --profiler PROFILER   To profile individual steps during training and assist in identifying bottlenecks.
  --benchmark [BENCHMARK]
                        If true enables cudnn.benchmark.
  --deterministic [DETERMINISTIC]
                        If true enables cudnn.deterministic.
  --reload_dataloaders_every_n_epochs RELOAD_DATALOADERS_EVERY_N_EPOCHS
                        Set to a non-negative integer to reload dataloaders every n epochs. Default: 0
  --reload_dataloaders_every_epoch [RELOAD_DATALOADERS_EVERY_EPOCH]
                        Set to True to reload dataloaders every epoch. .. deprecated:: v1.4 ``reload_dataloaders_every_epoch`` has been
                        deprecated in v1.4 and will be removed in v1.6. Please use ``reload_dataloaders_every_n_epochs``.
  --auto_lr_find [AUTO_LR_FIND]
                        If set to True, will make trainer.tune() run a learning rate finder, trying to optimize initial learning for
                        faster convergence. trainer.tune() method will set the suggested learning rate in self.lr or self.learning_rate
                        in the LightningModule. To use a different key set a string instead of True with the key name.
  --replace_sampler_ddp [REPLACE_SAMPLER_DDP]
                        Explicitly enables or disables sampler replacement. If not specified this will toggled automatically when DDP
                        is used. By default it will add ``shuffle=True`` for train sampler and ``shuffle=False`` for val/test sampler.
                        If you want to customize it, you can set ``replace_sampler_ddp=False`` and add your own distributed sampler.
  --terminate_on_nan [TERMINATE_ON_NAN]
                        If set to True, will terminate training (by raising a `ValueError`) at the end of each training batch, if any
                        of the parameters or the loss are NaN or +/-inf.
  --auto_scale_batch_size [AUTO_SCALE_BATCH_SIZE]
                        If set to True, will `initially` run a batch size finder trying to find the largest batch size that fits into
                        memory. The result will be stored in self.batch_size in the LightningModule. Additionally, can be set to either
                        `power` that estimates the batch size through a power search or `binsearch` that estimates the batch size
                        through a binary search.
  --prepare_data_per_node [PREPARE_DATA_PER_NODE]
                        If True, each LOCAL_RANK=0 will call prepare data. Otherwise only NODE_RANK=0, LOCAL_RANK=0 will prepare data
  --plugins PLUGINS     Plugins allow modification of core behavior like ddp and amp, and enable custom lightning plugins.
  --amp_backend AMP_BACKEND
                        The mixed precision backend to use ("native" or "apex")
  --amp_level AMP_LEVEL
                        The optimization level to use (O1, O2, etc...).
  --distributed_backend DISTRIBUTED_BACKEND
                        deprecated. Please use 'accelerator'
  --move_metrics_to_cpu [MOVE_METRICS_TO_CPU]
                        Whether to force internal logged metrics to be moved to cpu. This can save some gpu memory, but can make
                        training slower. Use with attention.
  --multiple_trainloader_mode MULTIPLE_TRAINLOADER_MODE
                        How to loop over the datasets when there are multiple train loaders. In 'max_size_cycle' mode, the trainer ends
                        one epoch when the largest dataset is traversed, and smaller datasets reload when running out of their data. In
                        'min_size' mode, all the datasets reload when reaching the minimum length of datasets.
  --stochastic_weight_avg [STOCHASTIC_WEIGHT_AVG]
                        Whether to use `Stochastic Weight Averaging (SWA) <https://pytorch.org/blog/pytorch-1.6-now-includes-
                        stochastic-weight-averaging/>_`
```
