import numpy as np

from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

def stratified_split(dataset, val_size=0.2, seed=42):
    labels = dataset.labels.numpy()
    indices = np.arange(len(dataset))

    train_idx, val_idx = train_test_split(indices, test_size=val_size, stratify=labels, random_state=seed)

    return Subset(dataset, train_idx), Subset(dataset, val_idx)