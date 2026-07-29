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


def get_stage_class_weights_from_dataframe(train_labels):
    # Stage 1: Missing vs Contact, over ALL training samples
    is_contact = (train_labels != 0).astype(int)
    counts1 = np.bincount(is_contact, minlength=2)
    alpha_stage1 = torch.tensor(counts1.sum() / (2 * counts1), dtype=torch.float32)

    # Stage 2: among CONTACT samples only
    contact_labels = train_labels[train_labels != 0] - 1  # shift to 0..6
    counts2 = np.bincount(contact_labels, minlength=7)
    alpha_stage2 = torch.tensor(counts2.sum() / (7 * counts2), dtype=torch.float32)

    return alpha_stage1, alpha_stage2