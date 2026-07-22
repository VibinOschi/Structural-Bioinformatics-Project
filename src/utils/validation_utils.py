import torch
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, validation_dataloader, training_history, label_encoder, device):
    model.eval()
    all_predicted, all_targets = [], []

    with torch.no_grad():
        for samples_batch, labels_batch in validation_dataloader:
            samples_batch, labels_batch = samples_batch.to(device), labels_batch.to(device)
            outputs = model(samples_batch)
            predicted = outputs.argmax(dim=1)
            all_predicted.append(predicted)
            all_targets.append(labels_batch)

    all_predicted = torch.cat(all_predicted).cpu().numpy()
    all_targets = torch.cat(all_targets).cpu().numpy()

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

    ax[1].plot(training_history['train_acc'], label='Train')
    ax[1].plot(training_history['val_acc'], label='Val')
    ax[1].set_title('Train & Val Accuracy')
    ax[1].set_xlabel('Epoch')
    ax[1].legend()

    plt.tight_layout()
    plt.show()

    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_predicted)
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.show()