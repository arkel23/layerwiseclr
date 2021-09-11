import torch.nn as nn
    
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
                nn.GELU(),
                nn.LayerNorm(hidden_size, eps=layer_norm_eps),
                nn.Linear(hidden_size, out_features) 
            )
        elif no_layers == 3:
            self.projection_head = nn.Sequential(
                #Flatten(),
                nn.Linear(in_features, hidden_size),
                nn.GELU(),
                nn.LayerNorm(hidden_size, eps=layer_norm_eps),
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.LayerNorm(hidden_size, eps=layer_norm_eps),
                nn.Linear(hidden_size, out_features)
            )

    def forward(self, x):
        return self.projection_head(x)
    

class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        return input_tensor.view(input_tensor.size(0), -1)
