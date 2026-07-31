import json
import os
import joblib

import torch
import argparse

from torch.utils.data import DataLoader
from xgboost import XGBClassifier

from src.utils.pdb.pdb_utils import get_extracted_features_from_directory_of_pdb
from src.utils.inference_utils import load_predictor, run_inference
from src.FeatureDataset import FeatureDataset
from train_xgboost import build_feature_matrix


def get_config():
    with open("configuration.json", "r") as file:
        configuration = json.load(file)

    return configuration


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-xgboost', help='Sets which model is used for prediction of contacts [True (default): XGBoost | False: Custom Model]', default=True)
    return parser.parse_args()


if __name__ == '__main__':
    config = get_config()
    args = arg_parser()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(config['prediction_output_directory'], exist_ok=True)

    pdb_extracted_features = get_extracted_features_from_directory_of_pdb(config)

    if args.xgboost is True:
        model = XGBClassifier()
        model.load_model(os.path.join(config['xgboost_output_dir'], 'xgboost_model.json'))

        components = joblib.load(os.path.join(config['xgboost_output_dir'], 'preprocessing_components.joblib'))
        feature_enc = components['feature_encoders']
        le = components['label_encoder']

        for pdb in pdb_extracted_features:
            samples, _ = build_feature_matrix(pdb, config['feature_columns'], feature_encoders=feature_enc, fit=False)
            predictions = model.predict(samples)
            probabilities = model.predict_proba(samples)

            prediction_labels = le.inverse_transform(predictions)
            confidence_scores = probabilities.max(axis=1)

            pdb['contact_predicted'] = prediction_labels
            pdb['score'] = confidence_scores

            pdb.to_csv(os.path.join(config['prediction_output_directory'], pdb['pdb_id'][0] + '.tsv'), sep='\t', index=False)

    else:
        model, label_encoder, feature_encoders = load_predictor(config, device)

        for pdb in pdb_extracted_features:
            dataset = FeatureDataset(pdb, config['feature_columns'], feature_encoders=feature_encoders, fit=False)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

            contact_pred, score = run_inference(model, dataloader, label_encoder, device)

            pdb['contact_predicted'] = contact_pred
            pdb['score'] = score

            pdb.to_csv(os.path.join(config['prediction_output_directory'], pdb['pdb_id'][0] + '.tsv'), sep='\t', index=False)