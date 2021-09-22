import os
import pandas as pd
from PIL import Image
from PIL import ImageFile

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder

from pytorch_lightning import LightningDataModule
from timm.data import create_transform

ImageFile.LOAD_TRUNCATED_IMAGES = True

def standard_transform(split, args):
    if split == 'train':
        transform = transforms.Compose([
            transforms.Resize(args.image_size+32),
            transforms.RandomCrop((args.image_size, args.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, 
                contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

    return transform


def deit_transform(split, args):
    resize_im = args.image_size > 32
    if split == 'train':
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.image_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.image_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.image_size) # to maintain same ratio w.r.t. 224 images
        t.append(transforms.Resize(size, interpolation=3))
        t.append(transforms.CenterCrop(args.image_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean=[0.5, 0.5, 0.5],
									std=[0.5, 0.5, 0.5]))
    return transforms.Compose(t)


def simclr_transform(args):
    s = 1
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    transform = transforms.Compose([
            transforms.RandomResizedCrop(size=args.image_size),
            transforms.RandomHorizontalFlip(),  # with 0.5 probability
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=args.image_size//20*2+1, sigma=(0.1, 2.0))], p=0.5),
            # We blur the image 50% of the time using a Gaussian kernel. We randomly sample σ ∈ [0.1, 2.0], and the kernel size is set to be 10% of the image height/width.
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        
    return transform

                
class ApplyTransform:
    """
    A stochastic data augmentation module that transforms any given data example randomly
    resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """
    def __init__(self, split, args):
        self.split = split
        self.args = args

        if args.mode in ['simclr', 'simlwclr'] and self.split == 'train':
            self.mode = 'simclr_train'
        else:
            self.mode = 'default'

        self.transform = self.build_transform()
    
    def __call__(self, x):
        if self.mode == 'simclr_train':
            return self.transform(x), self.transform(x)
        else:
            return self.transform(x)

    def build_transform(self):
        if self.args.deit_recipe:
            transform = deit_transform(split=self.split, args=self.args)
        elif self.mode == 'simclr_train':
            transform = simclr_transform(args=self.args)
        else:
            transform = standard_transform(split=self.split, args=self.args)
        return transform


class CIFAR10DM(LightningDataModule):

    def __init__(self, args):
        super().__init__()
        if args.dataset_path:
            self.data_dir = args.dataset_path
        else:
            self.data_dir = 'data'
        self.batch_size = args.batch_size
        self.image_size = args.image_size
        self.num_workers = args.no_cpu_workers
        self.transform_train = ApplyTransform(split='train', args=args)
        self.transform_eval = ApplyTransform(split='val', args=args)
        
        self.persist = True if (args.gpus > 1) else False
                    
    def prepare_data(self):
        '''called only once and on 1 GPU'''
        # download data
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        '''called on each GPU separately - stage defines if we are at fit or test step'''
        # we set up only relevant datasets when stage is specified (automatically set by Pytorch-Lightning)
        if stage == 'fit' or stage is None:
            dataset_train = CIFAR10(self.data_dir, train=True, transform=self.transform_train)
    
            no_train = int(len(dataset_train) * 0.95)
            no_val = len(dataset_train) - no_train

            self.dataset_train, self.dataset_val = random_split(dataset_train, [no_train, no_val])
            self.num_classes = len(dataset_train.classes)
            
        if stage == 'test' or stage is None:
            self.dataset_test = CIFAR10(self.data_dir, train=False, transform=self.transform_eval)
            self.num_classes = len(self.dataset_test.classes)

    def train_dataloader(self):
        '''returns training dataloader'''
        return DataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, persistent_workers=self.persist, drop_last=True)
        
    def val_dataloader(self):
        '''returns validation dataloader'''
        return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, persistent_workers=self.persist)
        
    def test_dataloader(self):
        '''returns test dataloader'''
        return DataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, persistent_workers=self.persist)
        

class CIFAR100DM(LightningDataModule):

    def __init__(self, args):
        super().__init__()
        if args.dataset_path:
            self.data_dir = args.dataset_path
        else:
            self.data_dir = 'data'
        self.batch_size = args.batch_size
        self.image_size = args.image_size
        self.num_workers = args.no_cpu_workers
        self.transform_train = ApplyTransform(split='train', args=args)
        self.transform_eval = ApplyTransform(split='val', args=args)
        
        self.persist = True if (args.gpus > 1) else False
        
    def prepare_data(self):
        '''called only once and on 1 GPU'''
        # download data
        CIFAR100(self.data_dir, train=True, download=True)
        CIFAR100(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        '''called on each GPU separately - stage defines if we are at fit or test step'''
        # we set up only relevant datasets when stage is specified (automatically set by Pytorch-Lightning)
        if stage == 'fit' or stage is None:
            dataset_train = CIFAR100(self.data_dir, train=True, transform=self.transform_train)
    
            no_train = int(len(dataset_train) * 0.95)
            no_val = len(dataset_train) - no_train

            self.dataset_train, self.dataset_val = random_split(dataset_train, [no_train, no_val])
            self.num_classes = len(dataset_train.classes)
            
        if stage == 'test' or stage is None:
            self.dataset_test = CIFAR100(self.data_dir, train=False, transform=self.transform_eval)
            self.num_classes = len(self.dataset_test.classes)

    def train_dataloader(self):
        '''returns training dataloader'''
        return DataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, persistent_workers=self.persist, drop_last=True)
        
    def val_dataloader(self):
        '''returns validation dataloader'''
        return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, persistent_workers=self.persist)
        
    def test_dataloader(self):
        '''returns test dataloader'''
        return DataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, persistent_workers=self.persist)   
         
            
class ImageNetDM(LightningDataModule):

    def __init__(self, args):
        super().__init__()
        self.data_dir = os.path.abspath(args.dataset_path)
        self.batch_size = args.batch_size
        self.image_size = args.image_size
        self.num_workers = args.no_cpu_workers
        self.transform_train = ApplyTransform(split='train', args=args)
        self.transform_eval = ApplyTransform(split='val', args=args)
        
        self.persist = True if (args.gpus > 1) else False
        
    def setup(self, stage=None):
        '''called on each GPU separately - stage defines if we are at fit or test step'''
        # we set up only relevant datasets when stage is specified (automatically set by Pytorch-Lightning)
        if stage == 'fit' or stage is None:
            data_path = os.path.join(self.data_dir, 'train')
            dataset_train = ImageFolder(root=data_path, transform=self.transform_train)
            
            no_train = int(len(dataset_train) * 0.99)
            no_val = len(dataset_train) - no_train
            
            self.dataset_train, self.dataset_val = random_split(dataset_train, [no_train, no_val])
            self.num_classes = len(dataset_train.classes)
        
        if stage == 'test' or stage is None:
            data_path = os.path.join(self.data_dir, 'val')
            self.dataset_test = ImageFolder(root=data_path, transform=self.transform_eval)
            self.num_classes = len(self.dataset_test.classes)
            
    def train_dataloader(self):
        '''returns training dataloader'''
        return DataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, persistent_workers=self.persist, drop_last=True)
        
    def val_dataloader(self):
        '''returns validation dataloader'''
        return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, persistent_workers=self.persist)
        
    def test_dataloader(self):
        '''returns test dataloader'''
        return DataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, persistent_workers=self.persist)
        

class DanbooruFacesFullDM(LightningDataModule):
    '''
	https://github.com/arkel23/Danbooru2018AnimeCharacterRecognitionDataset_Revamped
	'''
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.batch_size = args.batch_size
        self.num_workers = args.no_cpu_workers
        self.transform_train = ApplyTransform(split='train', args=args)
        self.transform_eval = ApplyTransform(split='val', args=args)
        
        self.persist = True if (args.gpus > 1) else False
        
    def setup(self, stage=None):
        '''called on each GPU separately - stage defines if we are at fit or test step'''
        # we set up only relevant datasets when stage is specified (automatically set by Pytorch-Lightning)
        if stage == 'fit' or stage is None:
            self.dataset_train = DanbooruFacesFull(self.args, split='train', transform=self.transform_train)
            self.dataset_val = DanbooruFacesFull(self.args, split='val', transform=self.transform_eval)
            
            self.num_classes = self.dataset_train.num_classes
        
        if stage == 'test' or stage is None:
            self.dataset_test = DanbooruFacesFull(self.args, split='test', transform=self.transform_eval)
            self.num_classes = self.dataset_test.num_classes
            
    def train_dataloader(self):
        '''returns training dataloader'''
        return DataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, persistent_workers=self.persist, drop_last=True)
        
    def val_dataloader(self):
        '''returns validation dataloader'''
        return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, persistent_workers=self.persist)
        
    def test_dataloader(self):
        '''returns test dataloader'''
        return DataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, persistent_workers=self.persist)
    

class DanbooruFacesFull(Dataset):
	'''
	https://github.com/arkel23/Danbooru2018AnimeCharacterRecognitionDataset_Revamped
	'''
	def __init__(self, args, 
	split='train', transform=None):
		super().__init__()
		self.dataset_name = args.dataset_name
		self.root = os.path.abspath(args.dataset_path)
		self.image_size = args.image_size
		self.split = split
		self.transform = transform

		if self.split=='train':
			print('Train set')
			self.set_dir = os.path.join(self.root, 'labels', 'train.csv')
			if self.transform is None:
				self.transform = ApplyTransform(split='train', args=args)

		elif self.split=='val':
			print('Validation set')
			self.set_dir = os.path.join(self.root, 'labels', 'val.csv')
			if self.transform is None:
				self.transform = ApplyTransform(split='val', args=args)

		else:
			print('Test set')
			self.set_dir = os.path.join(self.root, 'labels', 'test.csv')
			if self.transform is None:
				self.transform = ApplyTransform(split='test', args=args)
    
		self.df = pd.read_csv(self.set_dir, sep=',', header=None, names=['class_id', 'dir'], 
				dtype={'class_id': 'UInt16', 'dir': 'object'})
		
		self.targets = self.df['class_id'].to_numpy()
		self.data = self.df['dir'].to_numpy()
		
		self.classes = pd.read_csv(os.path.join(self.root, 'labels', 'classid_classname.csv'), 
		sep=',', header=None, names=['class_id', 'class_name'], 
		dtype={'class_id': 'UInt16', 'class_name': 'object'})
		self.num_classes = len(self.classes)
		
	def __getitem__(self, idx):
		
		if torch.is_tensor(idx):
			idx = idx.tolist()

		img_dir, target = self.data[idx], self.targets[idx]
		if self.dataset_name == 'danboorufaces':
			img_dir = os.path.join(self.root, 'faces', img_dir)
		elif self.dataset_name == 'danboorufull':
			img_dir = os.path.join(self.root, 'fullMin256', img_dir)
		img = Image.open(img_dir)

		if self.transform:
			img = self.transform(img)

		return img, target

	def __len__(self):
		return len(self.targets)