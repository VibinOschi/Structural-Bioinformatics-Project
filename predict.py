import json
import torch

from torch.utils.data import DataLoader

from src.utils.pdb.pdb_utils import get_extracted_features_from_directory_of_pdb
from src.utils.inference_utils import load_predictor, run_inference
from src.FeatureDataset import FeatureDataset

def get_config():
    with open("configuration.json", "r") as file:
        configuration = json.load(file)

    return configuration


if __name__ == '__main__':
    config = get_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pdb_extracted_features = get_extracted_features_from_directory_of_pdb(config)

    # TODO: initialize model (this is a bit incomplete) and for each protein inside the pd_extracted_features, run inference with the trained model
    model, label_encoder, feature_encoders = load_predictor(config, device)

    for pdb in pdb_extracted_features:
        dataset = FeatureDataset(pdb, config['feature_columns'], feature_encoders=feature_encoders, fit=False)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        var_1, var_2 = run_inference(model, dataloader, label_encoder, device)
        print(var_1)
        print(var_2)

    # TODO: Create the final output given the requirements of the project specifications
    # Output: a tsv that return a table with the following information
    #
    # Source residue columns (chain, index, insertion code, name)
    # Target residue columns (chain, index, insertion code, name)
    # Feature columns (...)
    # Interaction (predictor result)
    # Score