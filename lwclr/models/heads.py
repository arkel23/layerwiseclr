import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from .custom_losses import NT_XentSimCLR, SupConLoss
# https://github.com/PatrickHua/SimSiam/blob/main/models/simsiam.py

class MultiScaleToSingleScaleHead(nn.Module):
    def __init__(self, args, model, distill=False, detach=True):
        super().__init__()
        
        self.detach = detach
        
        if distill:
            model_name = args.model_name_teacher
        else:
            model_name = args.model_name
        
        original_dimensions = self.get_reduction_dims(model, args.image_size)
        final_dim = original_dimensions[-1]
        
        if model_name in ['alexnet', 'resnet18', 'resnet50', 'cifar_resnet18', 
            'resnet20', 'resnet56', 'resnet110', 'resnet8x4', 'resnet32x4']:            
            self.rescaling_head = nn.ModuleList([
                ProjectionMLPHead(batch_norm=args.bn_proj, no_layers=1, in_features=original_dim, out_features=final_dim)
                #nn.Linear(original_dim, final_dim)     
                for original_dim in original_dimensions])
        else:
            self.rescaling_head = nn.ModuleList([
                nn.Identity() for _ in original_dimensions])
            
    def get_reduction_dims(self, model, image_size):
        img = torch.rand(2, 3, image_size, image_size)
        features = model(img)
        dims = [layer_output.size(1) for layer_output in features]
        return dims
    
    def forward(self, x):
        if self.detach:
            interm_feats = [self.rescaling_head[i](features.detach()) for i, features in enumerate(x)]
        else:
            interm_feats = [self.rescaling_head[i](features) for i, features in enumerate(x)]
        return interm_feats


class ProjectionMLPHead(nn.Module):
    def __init__(self, linear: bool = False, batch_norm: bool = False, 
                 no_layers: int = 3, in_features: int = None, 
                 out_features: int = None, hidden_size: int = None, 
                 layer_norm_eps: float = 1e-12, dropout_prob: float = 0.1):
        super().__init__()
        
        self.no_layers = no_layers

        if no_layers != 1 and not hidden_size:
            hidden_size = out_features
        
        if linear:
            self.no_layers = 1
            self.projector = nn.Sequential(
                nn.Linear(in_features, out_features, bias=True)
            )
        else:
            if batch_norm:
                if no_layers == 1:
                    self.projector = nn.Sequential(
                        nn.Linear(in_features, out_features, bias=True),
                        nn.BatchNorm1d(out_features)
                    )
                elif no_layers == 2:
                    self.projector = nn.Sequential(
                        nn.Linear(in_features, hidden_size),
                        nn.BatchNorm1d(hidden_size),
                        nn.ReLU(inplace=True),
                        nn.Linear(hidden_size, out_features),
                        nn.BatchNorm1d(out_features)
                    )
                else:
                    self.projector = nn.Sequential(
                        nn.Linear(in_features, hidden_size),
                        nn.BatchNorm1d(hidden_size),
                        nn.ReLU(inplace=True),
                        nn.Linear(hidden_size, hidden_size),
                        nn.BatchNorm1d(hidden_size),
                        nn.ReLU(inplace=True),
                        nn.Linear(hidden_size, out_features),
                        nn.BatchNorm1d(out_features)
                    )
            else:
                if no_layers == 1:
                    self.projector = nn.Sequential(
                        nn.Linear(in_features, out_features, bias=True),
                        nn.LayerNorm(out_features, eps=layer_norm_eps)
                    )
                elif no_layers == 2:
                    self.projector = nn.Sequential(
                        nn.Linear(in_features, hidden_size),
                        nn.LayerNorm(hidden_size, eps=layer_norm_eps),
                        nn.GELU(),
                        nn.Linear(hidden_size, out_features),
                        nn.LayerNorm(out_features, eps=layer_norm_eps)
                    )
                else:
                    self.projector = nn.Sequential(
                        nn.Linear(in_features, hidden_size),
                        nn.LayerNorm(hidden_size, eps=layer_norm_eps),
                        nn.GELU(),
                        nn.Linear(hidden_size, hidden_size),
                        nn.LayerNorm(hidden_size, eps=layer_norm_eps),
                        nn.GELU(),
                        nn.Linear(hidden_size, out_features),
                        nn.LayerNorm(out_features, eps=layer_norm_eps)
                    )
                
    def forward(self, x):
        return self.projector(x)
        

class PredictionMLPHead(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048): # bottleneck structure
        super().__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h’s input and output (z and p) is d = 2048, 
        and h’s hidden layer’s dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x 


class LWContrastiveHead(nn.Module):
    def __init__(self, in_features: int, out_features: int, 
                hidden_size: int, no_layers: int = 2, 
                bn_proj: bool = False, temp: float = 0.5):
        super(LWContrastiveHead, self).__init__()
        
        self.bn_proj = bn_proj
        
        self.projector = ProjectionMLPHead(batch_norm=bn_proj, no_layers=no_layers,
                            in_features=in_features, hidden_size=hidden_size, out_features=out_features)
        
        self.criterion = SupConLoss(temperature=temp, base_temperature=temp, contrast_mode='all')
        
    def forward(self, features, features_aux):
        if self.bn_proj:
            z_feat_aux = [self.projector(feat.detach()) for feat in features_aux]
            z_feat = self.projector(features)
            z = torch.cat([z_feat, torch.stack(z_feat_aux, dim=1)], dim=1)
        else:    
            features_aux = torch.stack(features_aux, dim=1).detach()
            features = torch.cat([features.unsqueeze(1), features_aux], dim=1)
            z = self.projector(features)
            
        loss = self.criterion(F.normalize(z, dim=2))
        return loss