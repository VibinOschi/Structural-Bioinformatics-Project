import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef, roc_auc_score, roc_curve, auc as auc_fn
from sklearn.preprocessing import label_binarize

from src.utils.training_utils import predict_labels


def evaluate_model(model, validation_dataloader, training_history, label_encoder, device, output_dir):
    model.eval()
    all_predicted, all_targets, all_probs = [], [], []

    with torch.no_grad():
        for samples_batch_s, samples_batch_t, labels_batch in validation_dataloader:
            samples_batch_s, samples_batch_t, labels_batch = samples_batch_s.to(device), samples_batch_t.to(device), labels_batch.to(device)
            contact_logits, class_logits = model(samples_batch_s, samples_batch_t)
            predicted = predict_labels(contact_logits, class_logits)
            all_predicted.append(predicted)
            all_targets.append(labels_batch)

            contact_probs = torch.softmax(contact_logits, dim=1)
            class_probs = torch.softmax(class_logits, dim=1)

            no_contact_prob = contact_probs[:, 0:1]
            contact_prob = contact_probs[:, 1:2]
            weighted_class_probs = contact_prob * class_probs

            combined_probs = torch.cat([no_contact_prob, weighted_class_probs], dim=1)
            all_probs.append(combined_probs)

    all_predicted = torch.cat(all_predicted).cpu().numpy()
    all_targets = torch.cat(all_targets).cpu().numpy()
    all_probs = torch.cat(all_probs).cpu().numpy()

    class_names = list(label_encoder.classes_)

    print("Classification Report")
    print("=" * 60)
    print(classification_report(all_targets, all_predicted, target_names=class_names))

    # Graphs
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(training_history['train_loss'], label='Train')
    ax[0].plot(training_history['val_loss'], label='Val')
    ax[0].set_title('Train & Val Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].legend()
    ax[1].plot(training_history['train_f1'], label='Train')
    ax[1].plot(training_history['val_f1'], label='Val')
    ax[1].set_title('Train & F1 Score')
    ax[1].set_xlabel('Epoch')
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_f1_curves.png"), dpi=300, bbox_inches="tight")
    plt.show()

    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_predicted)
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300, bbox_inches="tight")
    plt.show()

    # Matthews correlation coefficient
    mcc = matthews_corrcoef(all_targets, all_predicted)
    print(f"Matthew's Correlation Coefficient: {mcc:.4f}")

    # ROC AUC
    auc = roc_auc_score(all_targets, all_probs, multi_class='ovr', average='macro')
    print(f"ROC AUC Score: {auc:.4f}")

    # ROC Curve (one-vs-rest per class)
    n_classes = len(class_names)
    targets_binarized = label_binarize(all_targets, classes=list(range(n_classes)))

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
    plt.savefig(os.path.join(output_dir, "roc_curves.png"), dpi=300, bbox_inches="tight")
    plt.show()