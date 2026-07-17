import torch
import pandas as pd

from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

class FeatureDataset(Dataset):
    def __init__(self, source_dataframe, feature_columns, label_column):
        # Features are maintained numerical if they already are, or they are categorically encoded otherwise
        self.encoders = {}
        feature_arrays = []

        for col in feature_columns:
            series = source_dataframe[col]

            if pd.api.types.is_numeric_dtype(series):
                feature_arrays.append(series.values.astype('float32'))
            else:
                le = LabelEncoder()
                encoded = le.fit_transform(series.astype(str))
                self.encoders[col] = le
                feature_arrays.append(encoded.astype('float32'))

        self.features = torch.tensor(pd.DataFrame(dict(zip(feature_columns, feature_arrays))).values, dtype=torch.float32)

        # Labels from 'Text Category' to 'Encoded Category'
        self.label_encoder = LabelEncoder()
        encoded_labels = self.label_encoder.fit_transform(source_dataframe[label_column].astype(str))
        self.labels = torch.tensor(encoded_labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]



# Note: to decode the label back into the category -> self.label_encoder.inverse_transform()