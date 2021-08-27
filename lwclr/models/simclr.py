# https://github.com/Spijkervet/SimCLR/blob/04bcf2baa1fb5631a0a636825aabe469865ad8a9/simclr/simclr.py#L8
# https://github.com/PyTorchLightning/lightning-bolts/blob/47eb2aae677350159c9ec0dc8ccdb6eef4217fff/pl_bolts/models/self_supervised/simclr/simclr_module.py
import torch.nn as nn

class SimCLR(nn.Module):
    def __init__(self, encoder, projection_dim, n_features, ret_interm_repr=False):
        super(SimCLR, self).__init__()

        self.encoder = encoder

        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a GELU non-linearity.
        # Original one used a ReLU. Changed to GeLU. PL implmementation also adds BN layer.
        self.projector = nn.Sequential(
            nn.Linear(n_features, n_features, bias=False),
            nn.GELU(),
            nn.Linear(n_features, projection_dim, bias=False),
        )

        self.ret_interm_repr = ret_interm_repr

    def inference(self, x_i):
        if not self.ret_interm_repr:
            return self.encoder(x_i)[:, 0]
        return self.encoder(x_i)[-1][:, 0]

    def forward(self, x_i, x_j):
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)
        
        # use cls token for projection
        if not self.ret_interm_repr:
            z_i = self.projector(h_i[:, 0])
            z_j = self.projector(h_j[:, 0])
        else:
            z_i = self.projector(h_i[-1][:, 0])
            z_j = self.projector(h_j[-1][:, 0])
        return h_i, h_j, z_i, z_j


