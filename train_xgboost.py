import json
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.utils.input_preprocessing import get_label_encoder_from_dataframe, preprocess_data_files_from_path


def get_config():
    with open("configuration.json", "r") as file:
        return json.load(file)


def build_feature_matrix(source_dataframe, feature_columns, feature_encoders=None, fit=True):
    """Flat-array version of FeatureDataset's encoding logic — same rules,
    but no s/t split and no tensors, since XGBoost wants one 2D array."""
    feature_encoders = feature_encoders or {}
    feature_arrays = []

    for col in feature_columns:
        series = source_dataframe[col]

        if col[2:] in ('ss8', '3di_state'):
            if fit:
                encoder = OneHotEncoder(sparse_output=False, dtype=np.float32, handle_unknown='ignore')
                encoded = encoder.fit_transform(series.astype(str).values.reshape(-1, 1))
                feature_encoders[col] = encoder
            else:
                encoded = feature_encoders[col].transform(series.astype(str).values.reshape(-1, 1))

        elif col[2:] in ('phi', 'psi'):
            theta = series.astype(np.float32).values
            encoded = np.stack([np.sin(theta), np.cos(theta)], axis=1)

        else:
            values = series.astype(np.float32).values.reshape(-1, 1)
            if fit:
                scaler = StandardScaler()
                encoded = scaler.fit_transform(values)
                feature_encoders[col] = scaler
            else:
                encoded = feature_encoders[col].transform(values)

        feature_arrays.append(encoded)

    X = np.concatenate(feature_arrays, axis=1)
    return X, feature_encoders


if __name__ == "__main__":
    config = get_config()

    source_df = preprocess_data_files_from_path(config['dataset_path'], config['feature_columns'])
    le = get_label_encoder_from_dataframe(source_df[config['label_column']])
    labels = le.transform(source_df[config['label_column']].astype(str))

    train_idx, val_idx = train_test_split(np.arange(len(source_df)), test_size=config['validation_split'], stratify=labels, random_state=config['rand_seed'])
    train_df, val_df = source_df.iloc[train_idx], source_df.iloc[val_idx]
    labels_train, labels_val = labels[train_idx], labels[val_idx]

    X_train, feature_encoders = build_feature_matrix(train_df, config['feature_columns'], fit=True)
    X_val, _ = build_feature_matrix(val_df, config['feature_columns'], feature_encoders=feature_encoders, fit=False)

    # Per-class sample weights, same inverse-frequency spirit as get_class_weights_from_dataframe
    class_counts = np.bincount(labels_train, minlength=len(le.classes_))
    class_weight_map = class_counts.sum() / (len(le.classes_) * class_counts)
    sample_weight = class_weight_map[labels_train]

    model = XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.07,
        objective='multi:softmax',
        num_class=len(le.classes_),
        eval_metric='mlogloss',
        early_stopping_rounds=20,

        tree_method="hist",
        device="cuda",
    )

    model.fit(
        X_train, labels_train,
        sample_weight=sample_weight,
        eval_set=[(X_val, labels_val)],
        verbose=True,
    )

    y_pred = model.predict(X_val)

    print("Classification Report")
    print("=" * 60)
    print(classification_report(labels_val, y_pred, target_names=list(le.classes_)))
    print(confusion_matrix(labels_val, y_pred))