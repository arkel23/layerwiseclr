import torch.nn as nn
    
class ProjectionHead(nn.Module):
    def __init__(self, n_layers: int = 1, n_features: int = None, n_classes: int = None, 
                 hidden_size: int = None, layer_norm_eps: float = 1e-12, dropout_prob: float = 0.1):
        super().__init__()
        self.n_layers = n_layers
        self.n_features = n_features
        self.n_classes = n_classes
        self.hidden_size = hidden_size
        
        if n_layers == 1:
            self.projection_head = nn.Sequential(
                LinearHead(n_features, n_classes)
            )     
        elif n_layers == 2:
            assert hidden_size, "hidden_size can't be None and n_layers>1"
            self.projection_head = nn.Sequential(
                Flatten(),
                nn.Linear(n_features, hidden_size),
                nn.GELU(),
                nn.LayerNorm(hidden_size, eps=layer_norm_eps),
                nn.Linear(hidden_size, n_classes) 
            )

    def forward(self, x):
        return self.projection_head(x)


class LinearHead(nn.Module):
    # use linear classifier
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.block_forward = nn.Sequential(Flatten(), nn.Linear(n_features, n_classes, bias=True))

    def forward(self, x):
        logits = self.block_forward(x)
        return logits


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        return input_tensor.view(input_tensor.size(0), -1)