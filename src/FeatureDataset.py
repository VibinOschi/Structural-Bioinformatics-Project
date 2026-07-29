import torch
import numpy as np

from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class FeatureDataset(Dataset):
    def __init__(self, source_dataframe, feature_columns, label_column=None, label_encoder=None, feature_encoders=None, fit=True):
        self.feature_encoders = feature_encoders or {}
        feature_arrays = []

        for col in feature_columns:
            series = source_dataframe[col]

            # Encoding Categories
            if col[2:] in ('ss8', '3di_state'):
                if fit:
                    encoder = OneHotEncoder(sparse_output=False, dtype=np.float32, handle_unknown='ignore')
                    encoded = encoder.fit_transform(series.astype(str).values.reshape(-1, 1))
                    self.feature_encoders[col] = encoder
                else:
                    encoder = self.feature_encoders[col]
                    encoded = encoder.transform(series.astype(str).values.reshape(-1, 1))

            # Angles in radians
            elif col[2:] in ('phi', 'psi'):
                theta = series.astype(np.float32).values
                encoded = np.stack([np.sin(theta), np.cos(theta)], axis=1)

            # Rest of the float values
            else:
                values = series.astype(np.float32).values.reshape(-1, 1)
                if fit:
                    scaler = StandardScaler()
                    encoded = scaler.fit_transform(values)
                    self.feature_encoders[col] = scaler
                else:
                    scaler = self.feature_encoders[col]
                    encoded = scaler.transform(values)

            feature_arrays.append(encoded)

        # Splitting the features to go each to source and target
        half = len(feature_arrays) // 2
        s_features = np.concatenate(feature_arrays[:half], axis=1)
        t_features = np.concatenate(feature_arrays[half:], axis=1)
        self.residue_features_s = torch.tensor(s_features, dtype=torch.float32)
        self.residue_features_t = torch.tensor(t_features, dtype=torch.float32)

        if label_column is not None:
            encoded_labels = label_encoder.transform(source_dataframe[label_column].astype(str))
            self.labels = torch.tensor(encoded_labels, dtype=torch.long)
        else:
            self.labels = None

    def __len__(self):
        return len(self.residue_features_s)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.residue_features_s[idx], self.residue_features_t[idx], self.labels[idx]
        return self.residue_features_s[idx], self.residue_features_t[idx]