# https://github.com/Spijkervet/SimCLR/blob/04bcf2baa1fb5631a0a636825aabe469865ad8a9/simclr/simclr.py#L8
import torch.nn as nn

class SimCLR(nn.Module):
    def __init__(self, encoder, projection_dim, n_features):
        super(SimCLR, self).__init__()

        self.encoder = encoder

        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a GELU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(n_features, n_features, bias=False),
            nn.GELU(),
            nn.Linear(n_features, projection_dim, bias=False),
        )

    def forward(self, x_i, x_j):
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)
        
        # use cls token for projection
        z_i = self.projector(h_i[:, 0])
        z_j = self.projector(h_j[:, 0])
        return h_i, h_j, z_i, z_j
