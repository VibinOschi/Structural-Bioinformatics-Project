import torch
import json

from torch.utils.data import DataLoader

from src.FeatureDataset import FeatureDataset
from src.Predictor import Predictor
from src.utils.input_preprocessing import get_label_encoder_from_dataframe, preprocess_data_files_from_path
from src.utils.dataset_utils import stratified_split, get_class_weights_from_dataframe
from src.utils.training_utils import train_model
from src.utils.validation_utils import evaluate_model


def get_config():
    with open("configuration.json", "r") as file:
        configuration = json.load(file)

    return configuration

if __name__ == "__main__":
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #
    source_df = preprocess_data_files_from_path(config['dataset_path'], config['feature_columns'])
    le = get_label_encoder_from_dataframe(source_df[config['label_column']])
    dataset = FeatureDataset(
        source_dataframe=source_df,
        feature_columns=config['feature_columns'],
        label_column=config['label_column'],
        label_encoder=le
    )

    #
    train_dataset, val_dataset = stratified_split(dataset, config['validation_split'], seed=config['rand_seed'])

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    class_weights = get_class_weights_from_dataframe(source_df, config['label_column'], le)

    #
    model = Predictor(dropout=config['dropout']).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['number_of_epochs'])

    #
    model, train_history = train_model(
        model=model,
        training_dataloader=train_dataloader,
        validation_dataloader=val_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=config['number_of_epochs'],
        patience=config['early_stopping_patience'],
        device=device
    )

    #
    evaluate_model(
        model=model,
        validation_dataloader=val_dataloader,
        training_history=train_history,
        label_encoder=le,
        device=device
    )