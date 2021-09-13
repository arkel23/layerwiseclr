import torch.nn as nn
# https://github.com/PatrickHua/SimSiam/blob/main/models/simsiam.py

class ProjectionHead(nn.Module):
    def __init__(self, no_layers: int = 1, in_features: int = None, 
                 out_features: int = None, hidden_size: int = None, 
                 layer_norm_eps: float = 1e-12, dropout_prob: float = 0.1):
        super().__init__()
        
        if no_layers != 1 and not hidden_size:
            hidden_size = out_features
            
        if no_layers == 1:
            self.projection_head = nn.Sequential(
                #Flatten(), 
                nn.Linear(in_features, out_features, bias=True)
            )     
        elif no_layers == 2:
            self.projection_head = nn.Sequential(
                #Flatten(),
                nn.Linear(in_features, hidden_size),
                nn.LayerNorm(hidden_size, eps=layer_norm_eps),
                nn.GELU(),
                nn.Linear(hidden_size, out_features),
                nn.LayerNorm(out_features, eps=layer_norm_eps),
            )
        elif no_layers == 3:
            self.projection_head = nn.Sequential(
                #Flatten(),
                nn.Linear(in_features, hidden_size),
                nn.LayerNorm(hidden_size, eps=layer_norm_eps),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size, eps=layer_norm_eps),
                nn.GELU(),
                nn.Linear(hidden_size, out_features),
                nn.LayerNorm(out_features, eps=layer_norm_eps),
            )

    def forward(self, x):
        return self.projection_head(x)
    

class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        return input_tensor.view(input_tensor.size(0), -1)


class ProjectionMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048, num_layers=3):
        super().__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        '''
        self.num_layers = num_layers
        
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(hidden_dim)
        )
    
    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x 


class PredictionMLP(nn.Module):
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
