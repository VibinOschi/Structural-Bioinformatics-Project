import torch
import numpy as np

from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder


'''
class FeatureDataset(Dataset):
    def __init__(self, source_dataframe, feature_columns, label_column, label_encoder):
        feature_arrays = []

        for col in feature_columns:
            series = source_dataframe[col]

            # Letter Category to One-Hot
            if col[2:] == 'ss8':
                encoder = OneHotEncoder(sparse_output=False, dtype=np.float32)
                encoded = encoder.fit_transform(series.astype(str).values.reshape(-1, 1))

            # Radians angles to Sin/Cos
            elif col[2:] in ('phi', 'psi'):
                theta = series.astype(np.float32).values
                encoded = np.stack([np.sin(theta), np.cos(theta)], axis=1)

            # Numeric Category to One-Hot
            elif col[2:] == '3di_state':
                encoder = OneHotEncoder(sparse_output=False, dtype=np.float32)
                encoded = encoder.fit_transform(series.astype(str).values.reshape(-1, 1))

            # Remaining Real-valued features
            else:
                encoded = series.astype(np.float32).values.reshape(-1, 1)

            feature_arrays.append(encoded)

        # Splitting between s_residue and t_residue
        half = len(feature_arrays) // 2
        s_features = np.concatenate(feature_arrays[:half], axis=1)
        t_features = np.concatenate(feature_arrays[half:], axis=1)

        self.residue_features_s = torch.tensor(s_features, dtype=torch.float32)
        self.residue_features_t = torch.tensor(t_features, dtype=torch.float32)

        # Labels from 'Text Category' to 'Encoded Category'
        encoded_labels = label_encoder.fit_transform(source_dataframe[label_column].astype(str))
        self.labels = torch.tensor(encoded_labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.residue_features_s[idx], self.residue_features_t[idx], self.labels[idx]
'''











class FeatureDataset(Dataset):
    def __init__(self, source_dataframe, feature_columns, label_column=None, label_encoder=None, feature_encoders=None, fit=True):
        self.feature_encoders = feature_encoders or {}
        feature_arrays = []

        for col in feature_columns:
            series = source_dataframe[col]
            if col[2:] in ('ss8', '3di_state'):
                if fit:
                    encoder = OneHotEncoder(sparse_output=False, dtype=np.float32, handle_unknown='ignore')
                    encoded = encoder.fit_transform(series.astype(str).values.reshape(-1, 1))
                    self.feature_encoders[col] = encoder
                else:
                    encoder = self.feature_encoders[col]
                    encoded = encoder.transform(series.astype(str).values.reshape(-1, 1))
            elif col[2:] in ('phi', 'psi'):
                theta = series.astype(np.float32).values
                encoded = np.stack([np.sin(theta), np.cos(theta)], axis=1)
            else:
                encoded = series.astype(np.float32).values.reshape(-1, 1)
            feature_arrays.append(encoded)

        half = len(feature_arrays) // 2
        s_features = np.concatenate(feature_arrays[:half], axis=1)
        t_features = np.concatenate(feature_arrays[half:], axis=1)
        self.residue_features_s = torch.tensor(s_features, dtype=torch.float32)
        self.residue_features_t = torch.tensor(t_features, dtype=torch.float32)

        if label_column is not None:
            if fit:
                encoded_labels = label_encoder.fit_transform(source_dataframe[label_column].astype(str))
            else:
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