import numpy as np
import torch
import json

from src.FeatureDataset import FeatureDataset
from src.utils.input_preprocessing import preprocess_data_files_from_path


def get_config():
    with open("configuration.json", "r") as file:
        configuration = json.load(file)

    return configuration

if __name__ == "__main__":
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = FeatureDataset(
        source_dataframe=preprocess_data_files_from_path(config['dataset_path'], config['feature_columns']))