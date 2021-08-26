import torch 
import torch.nn as nn

import einops
from einops.layers.torch import Rearrange

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