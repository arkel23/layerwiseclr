from collections import namedtuple
import torch 
import torch.nn as nn
import torchvision.models as models
import torchvision.models.feature_extraction as feature_extraction
from einops.layers.torch import Rearrange

from pytorch_pretrained_vit import ViT, ViTConfigExtended, PRETRAINED_CONFIGS

from .resnet_cifar import cifar_resnet18
from .resnet_others import resnet20, resnet32, resnet56, resnet110, resnet8x4, resnet32x4

def load_state_dict(args, model, ckpt_path):
    if args.load_partial_mode:
        model.model.load_partial(weights_path=ckpt_path, 
            pretrained_image_size=model.configuration.pretrained_image_size, 
            pretrained_mode=args.load_partial_mode, verbose=True)
    else:
        state_dict = torch.load(ckpt_path, map_location=torch.device('cpu'))
            
        state_dict_cp = state_dict['state_dict'].copy()
        for key in state_dict['state_dict'].keys():
            if 'backbone' in key:
                state_dict_cp[key.replace('backbone.', '')] = state_dict_cp[key]
                del state_dict_cp[key]
        state_dict = state_dict_cp
            
        expected_missing_keys = []
                    
        '''
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
        '''
            
        for key in expected_missing_keys:
            state_dict.pop(key)
                        
        ret = model.load_state_dict(state_dict, strict=False)
        print('''Missing keys when loading pretrained weights: {}
                Expected missing keys: {}'''.format(ret.missing_keys, expected_missing_keys))
        print('Unexpected keys when loading pretrained weights: {}'.format(ret.unexpected_keys))
            
        print('Loaded from custom checkpoint.')

def load_model(args, ret_interm_repr=True, pretrained=False, distill=False, ckpt_path=False):
    # initiates model and loss
    if distill and args.model_name_teacher:
        model_name = args.model_name_teacher
    else:
        model_name = args.model_name
            
    if model_name == 'alexnet':
        model = AlexNet(args, ret_interm_repr=ret_interm_repr, pretrained=pretrained)
    elif 'resnet' in model_name:
        model = ResNet(args, ret_interm_repr=ret_interm_repr, pretrained=pretrained)
    else:
        model = VisionTransformer(args, ret_interm_repr=ret_interm_repr, pretrained=pretrained)
    
    if ckpt_path:
        load_state_dict(args, model, ckpt_path)
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


class AlexNet(nn.Module):
    def __init__(self, args, ret_interm_repr=True, pretrained=False):
        super(AlexNet, self).__init__()
        
        self.ret_interm_repr = ret_interm_repr
        
        model = models.alexnet(pretrained=pretrained, progress=True)
        # node_names for outputs after Conv2d+ReLU
        # train_nodes, eval_nodes = feature_extraction.get_graph_node_names()
        # features.1, features.4, features.7, features.9, features.11
        return_nodes = {
            'features.1': 'layerminus5',
            'features.4': 'layerminus4',
            'features.7': 'layerminus3',
            'features.9': 'layerminus2',
            'features.11': 'layerminus1'
        }
        self.model = feature_extraction.create_feature_extractor(model, return_nodes=return_nodes)
        
        original_dimensions = self.get_reduction_dims(image_size=args.image_size)
        final_dim = original_dimensions[-1]
        
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                    Rearrange('b c 1 1 -> b c')
                    )
        
        Config = namedtuple('Config', ['hidden_size', 'num_classes', 'num_hidden_layers'])
        self.configuration = Config(hidden_size=final_dim, 
                                num_classes=args.num_classes, num_hidden_layers=len(original_dimensions))
        
    def get_reduction_dims(self, image_size):
        img = torch.rand(2, 3, image_size, image_size)
        features = self.model(img)
        dims = [layer_output.size(1) for layer_output in features.values()]
        return dims
    
    def forward(self, x):
        interm_features = list(self.model(x).values())
        if self.ret_interm_repr:
            return [self.pool(feats) for feats in interm_features]
        return self.pool(interm_features[-1])
    

class ResNet(nn.Module):
    def __init__(self, args, ret_interm_repr=True, pretrained=False):
        super(ResNet, self).__init__()
        
        self.ret_interm_repr = ret_interm_repr
        
        model, return_nodes = self.select_model_nodes(model_name=args.model_name, pretrained=pretrained)
        self.model = feature_extraction.create_feature_extractor(model, return_nodes=return_nodes)

        original_dimensions = self.get_reduction_dims(image_size=args.image_size)
        final_dim = original_dimensions[-1]
        
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                    Rearrange('b c 1 1 -> b c')
                    )
        
        Config = namedtuple('Config', ['hidden_size', 'num_classes', 'num_hidden_layers'])
        self.configuration = Config(hidden_size=final_dim, 
                                num_classes=args.num_classes, num_hidden_layers=len(original_dimensions))
             
    def get_reduction_dims(self, image_size):
        img = torch.rand(2, 3, image_size, image_size)
        features = self.model(img)
        dims = [layer_output.size(1) for layer_output in features.values()]
        return dims
    
    def forward(self, x):
        interm_features = list(self.model(x).values())
        if self.ret_interm_repr:
            return [self.pool(feats) for feats in interm_features]
        return self.pool(interm_features[-1])
    
    def select_model_nodes(self, model_name, pretrained):
        # node_names for outputs after Conv2d+ReLU
        # train_nodes, eval_nodes = feature_extraction.get_graph_node_names(model)
        if model_name == 'cifar_resnet18':
            model = cifar_resnet18(pretrained=pretrained, progress=True)
            return_nodes = {
                'layer3.0.relu': 'layerminus8',
                'layer3.0.relu_1': 'layerminus7',
                'layer3.1.relu': 'layerminus6',
                'layer3.1.relu_1': 'layerminus5',
                'layer4.0.relu': 'layerminus4',
                'layer4.0.relu_1': 'layerminus3',
                'layer4.1.relu': 'layerminus2',
                'layer4.1.relu_1': 'layerminus1'
            }
        elif model_name == 'resnet18':
            model = models.resnet18(pretrained=pretrained, progress=True)
            return_nodes = {
                'layer3.0.relu': 'layerminus8',
                'layer3.0.relu_1': 'layerminus7',
                'layer3.1.relu': 'layerminus6',
                'layer3.1.relu_1': 'layerminus5',
                'layer4.0.relu': 'layerminus4',
                'layer4.0.relu_1': 'layerminus3',
                'layer4.1.relu': 'layerminus2',
                'layer4.1.relu_1': 'layerminus1'
            }
        elif model_name == 'resnet50':
            model = models.resnet50(pretrained=pretrained, progress=True)
            return_nodes = {
                'layer4.0.relu': 'layerminus9',
                'layer4.0.relu_1': 'layerminus8',
                'layer4.0.relu_2': 'layerminus7',
                'layer4.1.relu': 'layerminus6',
                'layer4.1.relu_1': 'layerminus5',
                'layer4.1.relu_2': 'layerminus4',
                'layer4.2.relu': 'layerminus3',
                'layer4.2.relu_1': 'layerminus2',
                'layer4.2.relu_2': 'layerminus1'
            }
        elif model_name == 'resnet20':
            model = resnet20()
            return_nodes = {
                'layer3.0.relu': 'layerminus6',
                'layer3.0.relu_7': 'layerminus5',
                'layer3.1.relu': 'layerminus4',
                'layer3.1.relu_8': 'layerminus3',
                'layer3.2.relu': 'layerminus2',
                'layer3.2.relu_9': 'layerminus1'
            }
        elif model_name == 'resnet32':
            model = resnet32()
            return_nodes =  {
                'layer3.0.relu': 'layerminus10',
                'layer3.0.relu_11': 'layerminus9',
                'layer3.1.relu': 'layerminus8',
                'layer3.1.relu_12': 'layerminus7',
                'layer3.2.relu': 'layerminus6',
                'layer3.2.relu_13': 'layerminus5',
                'layer3.3.relu': 'layerminus4',
                'layer3.3.relu_14': 'layerminus3',
                'layer3.4.relu': 'layerminus2',
                'layer3.4.relu_15': 'layerminus1'
            }
        elif model_name == 'resnet56':
            model = resnet56()
            return_nodes =  {
                'layer3.0.relu': 'layerminus18',
                'layer3.0.relu_19': 'layerminus17',
                'layer3.1.relu': 'layerminus16',
                'layer3.1.relu_20': 'layerminus15',
                'layer3.2.relu': 'layerminus14',
                'layer3.2.relu_21': 'layerminus13',
                'layer3.3.relu': 'layerminus12',
                'layer3.3.relu_22': 'layerminus11',
                'layer3.4.relu': 'layerminus10',
                'layer3.4.relu_23': 'layerminus9',
                'layer3.5.relu': 'layerminus8',
                'layer3.5.relu_24': 'layerminus7',
                'layer3.6.relu': 'layerminus6',
                'layer3.6.relu_25': 'layerminus5',
                'layer3.7.relu': 'layerminus4',
                'layer3.7.relu_26': 'layerminus3',
                'layer3.8.relu': 'layerminus2',
                'layer3.8.relu_27': 'layerminus1'
            }
        elif model_name == 'resnet110':
            model = resnet110()
            return_nodes =  {
                'layer3.9.relu': 'layerminus18',
                'layer3.9.relu_46': 'layerminus17',
                'layer3.10.relu': 'layerminus16',
                'layer3.10.relu_47': 'layerminus15',
                'layer3.11.relu': 'layerminus14',
                'layer3.11.relu_48': 'layerminus13',
                'layer3.12.relu': 'layerminus12',
                'layer3.12.relu_49': 'layerminus11',
                'layer3.13.relu': 'layerminus10',
                'layer3.13.relu_50': 'layerminus9',
                'layer3.14.relu': 'layerminus8',
                'layer3.14.relu_51': 'layerminus7',
                'layer3.15.relu': 'layerminus6',
                'layer3.15.relu_52': 'layerminus5',
                'layer3.16.relu': 'layerminus4',
                'layer3.16.relu_53': 'layerminus3',
                'layer3.17.relu': 'layerminus2',
                'layer3.17.relu_54': 'layerminus1'
            }
        elif model_name == 'resnet8x4':
            model = resnet8x4()
            return_nodes = {
                'layer2.0.relu': 'layerminus4',
                'layer2.0.relu_2': 'layerminus3',
                'layer3.0.relu': 'layerminus2',
                'layer3.0.relu_3': 'layerminus1'
            }
        elif model_name == 'resnet32x4':
            model = resnet32x4()
            return_nodes =  {
                'layer3.0.relu': 'layerminus10',
                'layer3.0.relu_11': 'layerminus9',
                'layer3.1.relu': 'layerminus8',
                'layer3.1.relu_12': 'layerminus7',
                'layer3.2.relu': 'layerminus6',
                'layer3.2.relu_13': 'layerminus5',
                'layer3.3.relu': 'layerminus4',
                'layer3.3.relu_14': 'layerminus3',
                'layer3.4.relu': 'layerminus2',
                'layer3.4.relu_15': 'layerminus1'
            }
        else:
            raise NotImplementedError
        
        return model, return_nodes
