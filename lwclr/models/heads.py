import torch
import torch.nn as nn
import torch.nn.functional as F
from .custom_losses import NT_XentSimCLR, SupConLoss
# https://github.com/PatrickHua/SimSiam/blob/main/models/simsiam.py

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
            self.layer1 = nn.Sequential(
                nn.Linear(in_features, out_features, bias=True)
            )
        else:
            if batch_norm:
                if no_layers == 1:
                    self.layer1 = nn.Sequential(
                        nn.Linear(in_features, out_features, bias=True),
                        nn.BatchNorm1d(out_features)
                    )
                else:
                    self.layer1 = nn.Sequential(
                        nn.Linear(in_features, hidden_size),
                        nn.BatchNorm1d(hidden_size),
                        nn.ReLU(inplace=True)
                    )
                    self.layer2 = nn.Sequential(
                        nn.Linear(hidden_size, hidden_size),
                        nn.BatchNorm1d(hidden_size),
                        nn.ReLU(inplace=True)
                    )
                    self.layer3 = nn.Sequential(
                        nn.Linear(hidden_size, out_features),
                        nn.BatchNorm1d(out_features)
                    )
            else:
                if no_layers == 1:
                    self.layer1 = nn.Sequential(
                        nn.Linear(in_features, out_features, bias=True),
                        nn.LayerNorm(out_features, eps=layer_norm_eps)
                    )
                else:
                    self.layer1 = nn.Sequential(
                        nn.Linear(in_features, hidden_size),
                        nn.LayerNorm(hidden_size, eps=layer_norm_eps),
                        nn.GELU(inplace=True)
                    )
                    self.layer2 = nn.Sequential(
                        nn.Linear(hidden_size, hidden_size),
                        nn.LayerNorm(hidden_size, eps=layer_norm_eps),
                        nn.GELU(inplace=True)
                    )
                    self.layer3 = nn.Sequential(
                        nn.Linear(hidden_size, out_features),
                        nn.LayerNorm(out_features, eps=layer_norm_eps)
                    )
                
    def forward(self, x):
        if self.no_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.no_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        elif self.no_layers == 1:
            x = self.layer1(x)    
        return x
        

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
    
    
class SimContrastiveHead(nn.Module):
    def __init__(self, in_features: int, out_features: int, 
                hidden_size: int, no_layers: int = 2, 
                bn_proj: bool = False, temp: float = 0.5):
        super(SimContrastiveHead, self).__init__()
        
        self.projector = ProjectionMLPHead(batch_norm=bn_proj, no_layers=no_layers,
                            in_features=in_features, hidden_size=hidden_size, out_features=out_features)
        
        self.criterion = NT_XentSimCLR(temp=temp)
        
    def forward(self, h_i, h_j):
        z_i = self.projector(h_i)
        z_j = self.projector(h_j)
        
        loss = self.criterion(z_i, z_j)
        return loss
    
        