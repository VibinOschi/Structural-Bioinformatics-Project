import numpy as np
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

def stratified_split(dataset, val_size=0.2, seed=42):
    labels = dataset.labels.numpy()
    indices = np.arange(len(dataset))

    train_idx, val_idx = train_test_split(indices, test_size=val_size, stratify=labels, random_state=seed)

    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def get_class_weights_from_dataframe(dataframe, label_column, label_encoder):
    class_counts = dataframe[label_column].value_counts()
    counts_ordered = torch.tensor([class_counts[cls] for cls in label_encoder.classes_], dtype=torch.float32)
    class_weights = torch.reciprocal(counts_ordered)
    class_weights = class_weights / class_weights.sum() * len(label_encoder.classes_)
    return class_weights