import torch
import torch.nn as nn

'''
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

        self.layers = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32, 8)
        )

        self._initialize_weights()

    def forward(self, x_s, x_t):
        embedding_s = self.encoder_s(x_s)
        embedding_t = self.encoder_t(x_t)
        combined_embeddings = torch.cat([embedding_s, embedding_t], dim=1)
        return self.layers(combined_embeddings)

    def _initialize_weights(self):
        for m in self.layers.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                nn.init.zeros_(m.bias)
'''

import torch.nn.functional as F


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

'''
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

        self.shared = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
        )

        # Stage 1: contact vs. Missing
        self.contact_head = nn.Linear(32, 1)
        # Stage 2: which of the 7 contact classes (only meaningful if contact=True)
        self.class_head = nn.Linear(32, 7)

        self._initialize_weights()

    def forward(self, x_s, x_t):
        embedding_s = self.encoder_s(x_s)
        embedding_t = self.encoder_t(x_t)
        combined_embeddings = torch.cat([embedding_s, embedding_t], dim=1)
        features = self.shared(combined_embeddings)

        contact_logit = self.contact_head(features).squeeze(-1)      # (B,)
        class_logits = self.class_head(features)                      # (B, 7)

        log_p_missing = F.logsigmoid(-contact_logit)                   # log P(Missing)
        log_p_contact = F.logsigmoid(contact_logit)                    # log P(contact)
        log_p_class_given_contact = F.log_softmax(class_logits, dim=1)  # log P(class | contact)

        # Assumes label index 0 = "Missing", indices 1..7 = contact classes.
        log_probs = torch.cat([
            log_p_missing.unsqueeze(1),
            log_p_contact.unsqueeze(1) + log_p_class_given_contact,
        ], dim=1)  # (B, 8), rows sum to 1 in probability space

        return log_probs

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                nn.init.zeros_(m.bias)
'''

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

        self.shared = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
        )

        # Stage 1: Missing (0) vs Contact (1) — 2 logits, reuse FocalLoss as-is
        self.contact_head = nn.Linear(32, 2)
        # Stage 2: which of the 7 contact classes
        self.class_head = nn.Linear(32, 7)

        self._initialize_weights()

    def forward(self, x_s, x_t):
        embedding_s = self.encoder_s(x_s)
        embedding_t = self.encoder_t(x_t)
        combined = torch.cat([embedding_s, embedding_t], dim=1)
        features = self.shared(combined)

        contact_logits = self.contact_head(features)  # (B, 2)
        class_logits = self.class_head(features)      # (B, 7)
        return contact_logits, class_logits

    def _initialize_weights(self):
        for module in (self.shared, self.contact_head, self.class_head):
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                    nn.init.zeros_(m.bias)