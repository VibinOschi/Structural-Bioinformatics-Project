import json
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, matthews_corrcoef, roc_auc_score, roc_curve, auc as auc_fn
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize
from xgboost import XGBClassifier

from src.utils.input_preprocessing import get_label_encoder_from_dataframe, preprocess_data_files_from_path


def get_config():
    with open("configuration.json", "r") as file:
        return json.load(file)


def build_feature_matrix(source_dataframe, feature_columns, feature_encoders=None, fit=True):
    # Feature matrix used by XGBoost, with every feature transformed in a similar way as the other model
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

    features = np.concatenate(feature_arrays, axis=1)
    return features, feature_encoders


if __name__ == "__main__":
    config = get_config()

    source_df = preprocess_data_files_from_path(config['dataset_path'], config['feature_columns'])
    le = get_label_encoder_from_dataframe(source_df[config['label_column']])
    labels = le.transform(source_df[config['label_column']].astype(str))

    train_idx, val_idx = train_test_split(np.arange(len(source_df)), test_size=config['validation_split'], stratify=labels, random_state=config['rand_seed'])
    train_df, val_df = source_df.iloc[train_idx], source_df.iloc[val_idx]
    labels_train, labels_val = labels[train_idx], labels[val_idx]

    samples_train, feature_enc = build_feature_matrix(train_df, config['feature_columns'], fit=True)
    samples_val, _ = build_feature_matrix(val_df, config['feature_columns'], feature_encoders=feature_enc, fit=False)

    # Per-class sample weights
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
        n_jobs=4,  # To use only in the case of CPU training (commenting the following two lines)
        # tree_method="hist",
        # device="cuda",
    )

    model.fit(
        samples_train, labels_train,
        sample_weight=sample_weight,
        eval_set=[(samples_val, labels_val)],
        verbose=True,
    )

    # Model saving
    output_dir = config['xgboost_output_dir']
    os.makedirs(output_dir, exist_ok=True)
    model.save_model(os.path.join(output_dir, "xgboost_model.json"))
    joblib.dump(
        {
            "feature_encoders": feature_enc,
            "label_encoder": le,
            "feature_columns": config['feature_columns'],
        },
        os.path.join(output_dir, "preprocessing_components.joblib"),
    )

    y_pred = model.predict(samples_val)
    all_probs = model.predict_proba(samples_val)

    # ~~~ Evaluation ~~~

    print("Classification Report")
    print("=" * 60)
    print(classification_report(labels_val, y_pred, target_names=list(le.classes_)))
    print(confusion_matrix(labels_val, y_pred))

    output_dir_eval = config['xgboost_output_dir'] + 'eval/'
    os.makedirs(output_dir_eval, exist_ok=True)

    # Confusion Matrix
    class_names = list(le.classes_)
    cm = confusion_matrix(labels_val, y_pred)

    fig_cm, ax_cm = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax_cm, cmap="Blues", xticks_rotation=45, colorbar=True)
    ax_cm.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_eval, "confusion_matrix.png"), dpi=300, bbox_inches="tight")
    plt.show()

    # Matthews correlation coefficient
    mcc = matthews_corrcoef(labels_val, y_pred)
    print(f"Matthew's Correlation Coefficient: {mcc:.4f}")

    # ROC AUC
    auc = roc_auc_score(labels_val, all_probs, multi_class='ovr', average='macro')
    print(f"ROC AUC Score: {auc:.4f}")

    # ROC Curve (one-vs-rest per class)
    class_names = list(le.classes_)
    n_classes = len(class_names)
    targets_binarized = label_binarize(labels_val, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(7, 6))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(targets_binarized[:, i], all_probs[:, i])
        class_auc = auc_fn(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {class_auc:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves (One-vs-Rest) — Macro AUC = {auc:.3f}")
    ax.legend(loc="lower right", fontsize="small")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_eval, "roc_curves.png"), dpi=300, bbox_inches="tight")
    plt.show()




    # --- Reference: how to load everything back for inference ---
    #
    # from xgboost import XGBClassifier
    # import joblib
    #
    # inference_model = XGBClassifier()
    # inference_model.load_model("xgb_model.json")
    #
    # artifacts = joblib.load("preprocessing_artifacts.joblib")
    # feature_encoders = artifacts["feature_encoders"]
    # le = artifacts["label_encoder"]
    # feature_columns = artifacts["feature_columns"]
    #
    # X_new, _ = build_feature_matrix(new_df, feature_columns, feature_encoders=feature_encoders, fit=False)
    # preds = inference_model.predict(X_new)
    # pred_labels = le.inverse_transform(preds)