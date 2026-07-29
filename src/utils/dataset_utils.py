import numpy as np
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

def stratified_split(dataset, val_size=0.2, seed=42):
    labels = dataset.labels.numpy()
    indices = np.arange(len(dataset))

    train_idx, val_idx = train_test_split(indices, test_size=val_size, stratify=labels, random_state=seed)

    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def get_class_weights_from_dataframe(train_labels, max_weight=20.0):
    # Stage 1 -> Missing or Contact
    is_contact = (train_labels != 0).astype(int)
    counts1 = np.bincount(is_contact, minlength=2)
    alpha_stage_1 = torch.tensor(counts1.sum() / (2 * counts1), dtype=torch.float32)

    # Stage 2 -> Among Contacts
    contact_labels = train_labels[train_labels != 0] - 1
    counts2 = np.bincount(contact_labels, minlength=7)

    alpha_stage_2 = counts2.sum() / (7 * counts2)
    alpha_stage_2 = np.clip(alpha_stage_2, a_min=None, a_max=max_weight)
    alpha_stage_2 = torch.tensor(alpha_stage_2, dtype=torch.float32)

    return alpha_stage_1, alpha_stage_2