import torch
import torch.nn as nn


class ResidueEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=32, dropout=0.0):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.LeakyReLU(),
        )
        self._initialize_weights()

    def forward(self, x):
        return self.layers(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                nn.init.zeros_(m.bias)


class Predictor(nn.Module):
    def __init__(self, dropout, embedding_dim=32, shared_encoder=True):
        super().__init__()

        if shared_encoder:
            encoder = ResidueEncoder(39, output_dim=embedding_dim, dropout=dropout)
            self.encoder_s = encoder
            self.encoder_t = encoder
        else:
            self.encoder_s = ResidueEncoder(39, output_dim=embedding_dim, dropout=dropout)
            self.encoder_t = ResidueEncoder(39, output_dim=embedding_dim, dropout=dropout)

        self.linear_layers = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
        )

        # Stage 1 -> Missing or Contacts
        self.contact_head = nn.Linear(32, 2)
        # Stage 2 -> Which Contact
        self.class_head = nn.Linear(32, 7)

        self._initialize_weights()

    def forward(self, x_s, x_t):
        embedding_s = self.encoder_s(x_s)
        embedding_t = self.encoder_t(x_t)
        combined = torch.cat([embedding_s, embedding_t], dim=1)
        features = self.linear_layers(combined)

        contact_logits = self.contact_head(features)
        class_logits = self.class_head(features)
        return contact_logits, class_logits

    def _initialize_weights(self):
        for module in (self.linear_layers, self.contact_head, self.class_head):
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                    nn.init.zeros_(m.bias)