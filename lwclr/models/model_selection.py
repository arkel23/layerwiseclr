from collections import namedtuple
import torch 
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange

import timm
from efficientnet_pytorch import EfficientNet
from pytorch_pretrained_vit import ViT, ViTConfigExtended, PRETRAINED_CONFIGS

def load_model(args, ret_interm_repr=True, pretrained=False):
    # initiates model and loss
    if args.model_name == 'effnet_b0':
        model = EfficientNet(args, ret_interm_repr=ret_interm_repr, pretrained=pretrained)
    elif 'resnet' in args.model_name:
        model = ResNet(args, ret_interm_repr=ret_interm_repr, pretrained=pretrained)
    else:
        model = VisionTransformer(args, ret_interm_repr=ret_interm_repr, pretrained=pretrained)
    
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
    def __init__(self, args, ret_interm_repr=True, pretrained=False):
        super(VisionTransformer, self).__init__()
        
        self.ret_interm_repr = ret_interm_repr
        
        def_config = PRETRAINED_CONFIGS['{}'.format(args.model_name)]['config']
        self.configuration = ViTConfigExtended(**def_config)
        self.configuration.image_size = args.image_size
        self.configuration.num_classes = args.num_classes
        
        self.model = ViT(self.configuration, name=args.model_name, 
            pretrained=pretrained, conv_patching=args.conv_patching, 
            ret_interm_repr=ret_interm_repr, load_fc_layer=False)
        
        if args.vit_avg_pooling:
            self.pool = nn.Sequential(
                Rearrange('b s d -> b d s'),
                nn.AdaptiveAvgPool1d(1),
                Rearrange('b d 1 -> b d')
            )
        else:
            self.norm = nn.LayerNorm(self.configuration.hidden_size, 
                                     eps=self.configuration.layer_norm_eps)
        
    def forward(self, images, mask=None):
        if self.ret_interm_repr:
            _, interm_features = self.model(images, mask)
            if hasattr(self, 'pool'):
                return [self.pool(features) for features in interm_features]
            return [self.norm(features[:, 0]) for features in interm_features]
        if hasattr(self, 'pool'):
            return self.pool(self.model(images, mask))
        return self.norm(self.model(images, mask)[:, 0])

class EffNet(nn.Module):
    def __init__(self, args, ret_interm_repr=True, pretrained=False):
        super(EffNet, self).__init__()
        
        self.ret_interm_repr = ret_interm_repr
        
        if pretrained:
            self.model = EfficientNet.from_pretrained('efficientnet-b0')
        else:
            self.model = EfficientNet.from_name('efficientnet-b0')

        if not pretrained:
            self.init_weights()
        
        original_dimensions = self.get_reduction_dims(image_size=args.image_size)
        final_dim = original_dimensions[-1]
        
        if self.ret_interm_repr:
            self.reshaping_head = nn.ModuleList([
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    Rearrange('b c 1 1 -> b c'),
                    nn.Linear(original_dim, final_dim)
                ) 
            for original_dim in original_dimensions])
        else:
            self.reshaping_head = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                    Rearrange('b c 1 1 -> b c')
            )
        
        Config = namedtuple('Config', ['hidden_size', 'num_classes', 'num_hidden_layers'])
        self.configuration = Config(hidden_size=final_dim, 
                                num_classes=args.num_classes, num_hidden_layers=len(original_dimensions))

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
    
    def get_reduction_dims(self, image_size):
        img = torch.rand(2, 3, image_size, image_size)
        features = self.model.extract_endpoints(img)
        dims = [layer_output.size(1) for layer_output in features.values()]
        return dims
        
    def forward(self, x):
        if self.ret_interm_repr:
            interm_features = self.model.extract_endpoints(x).values()
            return [self.reshaping_head[i](features) for i, features in enumerate(interm_features)]
        return self.reshaping_head(self.model.extract_features(x))


class ResNet(nn.Module):
    def __init__(self, args, ret_interm_repr=True, pretrained=False):
        super(ResNet, self).__init__()
        
        self.ret_interm_repr = ret_interm_repr
        
        self.model = timm.create_model('{}'.format(args.model_name), pretrained=pretrained, features_only=True)
        
        if not pretrained:
            self.init_weights()
        
        original_dimensions = self.get_reduction_dims(image_size=args.image_size)
        final_dim = original_dimensions[-1]
        
        if self.ret_interm_repr:
            self.reshaping_head = nn.ModuleList([
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    Rearrange('b c 1 1 -> b c'),
                    nn.Linear(original_dim, final_dim)
                ) 
            for original_dim in original_dimensions])
        else:
            self.reshaping_head = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                    Rearrange('b c 1 1 -> b c')
            )
        
        Config = namedtuple('Config', ['hidden_size', 'num_classes', 'num_hidden_layers'])
        self.configuration = Config(hidden_size=final_dim, 
                                num_classes=args.num_classes, num_hidden_layers=len(original_dimensions))
         
    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)
            
        self.apply(_init)
        
    def get_reduction_dims(self, image_size):
        img = torch.rand(2, 3, image_size, image_size)
        features = self.model(img)
        dims = [layer_output.size(1) for layer_output in features]
        return dims
    
    def forward(self, x):
        if self.ret_interm_repr:
            interm_features = self.model(x)
            return [self.reshaping_head[i](features) for i, features in enumerate(interm_features)]
        return self.reshaping_head(self.model(x)[-1])




