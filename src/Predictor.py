import torch.nn as nn

# TODO: find a better architecture

class Predictor(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(22, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32, 8)
        )
        self._initialize_weights()

    def forward(self, x):
        return self.layers(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                nn.init.zeros_(m.bias)





'''
import torch
import torch.nn as nn


class ResidueEncoder(nn.Module):
    """Shared-weight encoder applied identically to source and target residues."""

    def __init__(
        self,
        ss8_classes: int = 8,
        di_classes: int = 20,
        ss8_emb_dim: int = 8,
        di_emb_dim: int = 12,
        hidden: int = 64,
        dropout: float = 0.1,
        angles_in_degrees: bool = True,
    ):
        super().__init__()
        self.angles_in_degrees = angles_in_degrees
        self.ss8_emb = nn.Embedding(ss8_classes, ss8_emb_dim)
        self.di_emb = nn.Embedding(di_classes, di_emb_dim)

        cont_dim = 1 + 4 + 5  # rsa + sin/cos(phi,psi) + atchley
        in_dim = ss8_emb_dim + di_emb_dim + cont_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.LeakyReLU(),
        )
        self.out_dim = hidden

    def forward(self, ss8_idx, rsa, phi, psi, atchley, di_idx):
        ss8 = self.ss8_emb(ss8_idx)
        di = self.di_emb(di_idx)

        if self.angles_in_degrees:
            phi = torch.deg2rad(phi)
            psi = torch.deg2rad(psi)
        ang = torch.stack(
            [torch.sin(phi), torch.cos(phi), torch.sin(psi), torch.cos(psi)], dim=-1
        )

        x = torch.cat([ss8, di, rsa.unsqueeze(-1), ang, atchley], dim=-1)
        return self.net(x)


class PairInteraction(nn.Module):
    """concat(u, v, |u-v|, u*v). Swap for u+v instead of concat(u,v) if
    contact type is symmetric under source/target swap."""

    def forward(self, u, v):
        diff = torch.abs(u - v)
        prod = u * v
        return torch.cat([u, v, diff, prod], dim=-1)


class Predictor(nn.Module):
    def __init__(
        self,
        dropout: float = 0.1,
        ss8_classes: int = 8,
        di_classes: int = 20,
        n_classes: int = 8,
    ):
        super().__init__()
        self.encoder = ResidueEncoder(
            ss8_classes=ss8_classes, di_classes=di_classes, dropout=dropout
        )
        self.pair = PairInteraction()

        pair_dim = self.encoder.out_dim * 4
        self.classifier = nn.Sequential(
            nn.Linear(pair_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32, n_classes),
        )
        self._initialize_weights()

    def forward(
        self,
        s_ss8, s_rsa, s_phi, s_psi, s_atchley, s_di,
        t_ss8, t_rsa, t_phi, t_psi, t_atchley, t_di,
    ):
        u = self.encoder(s_ss8, s_rsa, s_phi, s_psi, s_atchley, s_di)
        v = self.encoder(t_ss8, t_rsa, t_phi, t_psi, t_atchley, t_di)
        pair = self.pair(u, v)
        return self.classifier(pair)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
                nn.init.zeros_(m.bias)
'''