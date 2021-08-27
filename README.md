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

Train a SimLWCLR model with batch size=256, with the first (0) and last (-1) layer for positive pairs and all previous settings:
```
python train.py --gpus 1 --image_size 128 --dataset_path data --max_epochs 300 --dataset_name cifar10 --mode simlwclr --batch_size 256 --save_checkpoint_freq 50 --layer_pair_1 0 --layer_pair_2 -1

```

Train a LWPLCLR (LayerWise PseudoLabels CLR) model:
```
python train.py --gpus 1 --image_size 128 --dataset_path data --max_epochs 300 --dataset_name cifar10 --mode lwplclr --batch_size 64 --save_checkpoint_freq 50
```

## Description of models
### SimCLR
Takes two augmentations from same image for positive pairs, and other images in batch for negative pairs.

### SimLWCLR
Takes two different layer representations from same image for positive pairs, and other images in batch for negative pairs.

### LWPLCLR
Same as SimCLR, but additionally generates pseudolabels for each image in batch (image 0 gets label 0, image n gets label n),
and also gives the same pseudolabel to each layer-wise representation of the image, and similarly for the other augmentation 
of the same image. Then concatenates all these representations and images across the batch dimension, and passes them through
a MLP classification head to predict which image is this representation from.
