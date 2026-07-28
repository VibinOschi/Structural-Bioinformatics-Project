import json
import torch

from src.utils.pdb.pdb_utils import get_extracted_features_from_directory_of_pdb

def get_config():
    with open("configuration.json", "r") as file:
        configuration = json.load(file)

    return configuration


# Output: a tsv that return a table with the following information
#
# Source residue columns (chain, index, insertion code, name)
# Target residue columns (chain, index, insertion code, name)
# Feature columns (...)
# Interaction (predictor result)
# Score 
if __name__ == '__main__':
    config = get_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pdb_extracted_features = get_extracted_features_from_directory_of_pdb(config)
