import torch 
import torch.nn as nn
import torchvision.models as models

import einops
from einops.layers.torch import Rearrange

from efficientnet_pytorch import EfficientNet
from pytorch_pretrained_vit import ViT, ViTConfigExtended, PRETRAINED_CONFIGS

def load_model(args, ret_interm_repr=True):
    # initiates model and loss     
    model = VisionTransformer(args, ret_interm_repr=ret_interm_repr)
    
    if args.checkpoint_path:
        if args.load_partial_mode:
            model.model.load_partial(weights_path=args.checkpoint_path, 
                pretrained_image_size=model.configuration.pretrained_image_size, 
                pretrained_mode=args.load_partial_mode, verbose=True)
        else:
            state_dict = torch.load(args.checkpoint_path, map_location=torch.device('cpu'))
            expected_missing_keys = []

            load_patch_embedding = (
                (model.configuration.num_channels==model.configuration.pretrained_num_channels) and
                (not(args.conv_patching))
            )
            
            if ('patch_embedding.weight' in state_dict and load_patch_embedding):
                expected_missing_keys += ['model.patch_embedding.weight', 'model.patch_embedding.bias']
            
            if ('pre_logits.weight' in state_dict and model.configuration.load_repr_layer==False):
                expected_missing_keys += ['model.pre_logits.weight', 'model.pre_logits.bias']
                    
            if ('model.fc.weight' in state_dict and model.config.load_fc_layer):
                expected_missing_keys += ['model.fc.weight', 'model.fc.bias']
            
            for key in expected_missing_keys:
                state_dict.pop(key)
                        
            ret = model.load_state_dict(state_dict, strict=False)
            print('''Missing keys when loading pretrained weights: {}
                Expected missing keys: {}'''.format(ret.missing_keys, expected_missing_keys))
            print('Unexpected keys when loading pretrained weights: {}'.format(ret.unexpected_keys))
            
            print('Loaded from custom checkpoint.')

    return model


class VisionTransformer(nn.Module):
    def __init__(self, args, ret_interm_repr=True):
        super(VisionTransformer, self).__init__()
        
        def_config = PRETRAINED_CONFIGS['{}'.format(args.model_name)]['config']
        self.configuration = ViTConfigExtended(**def_config)
        self.configuration.num_classes = args.num_classes
        self.configuration.image_size = args.image_size

        self.ret_interm_repr = ret_interm_repr
        
        self.model = ViT(self.configuration, name=args.model_name, 
            pretrained=args.pretrained_checkpoint, conv_patching=args.conv_patching, 
            ret_interm_repr=ret_interm_repr, load_fc_layer=False)

    def forward(self, images, mask=None):
        if self.ret_interm_repr:
            _, interm_features = self.model(images, mask)
            return interm_features
        return self.model(images, mask)


class ResNet(nn.Module):
    def __init__(self, args, ret_interm_repr):
        super(ResNet, self).__init__()
        
        if args.model_name == 'resnet18':
            base_model = models.resnet18(pretrained=args.pretrained, progress=True)
        elif args.model_name == 'resnet50':
            base_model = models.resnet50(pretrained=args.pretrained, progress=True) 
        elif args.model_name == 'resnet152':
            base_model = models.resnet152(pretrained=args.pretrained, progress=True)
        self.model = base_model

        # Initialize/freeze weights
        # originally for pretrained would freeze all layers except last
        #if args.pretrained:
        #    freeze_layers(self.model)
        #else:
        if not args.pretrained:
            self.init_weights()
        
        # Classifier head
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, args.num_classes)

    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)
            
        self.apply(_init)
        nn.init.constant_(self.model.fc.weight, 0)
        nn.init.constant_(self.model.fc.bias, 0)
        
    def forward(self, x):
        out = self.model(x)
        return out


class EffNet(nn.Module):
    def __init__(self, args, ret_interm_repr):
        super(EffNet, self).__init__()
        
        if args.pretrained:
            self.model = EfficientNet.from_pretrained('efficientnet-b0')
        else:
            self.model = EfficientNet.from_name('efficientnet-b0')

        if not args.pretrained:
            self.init_weights()
        
        # Classifier head
        num_features = self.model._fc.in_features
        self.model._fc = nn.Linear(num_features, args.num_classes)

        self.ret_interm_repr = ret_interm_repr

    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)
            
        self.apply(_init)
        nn.init.constant_(self.model._fc.weight, 0)
        nn.init.constant_(self.model._fc.bias, 0)
        
    def forward(self, x):
        if self.ret_interm_repr:
            return self.model.extract_endpoints(img)
        return self.model.extract_features(x)

