import torch
import pickle

from pathlib import Path
from sklearn.metrics import f1_score


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        self.register_buffer('alpha', alpha if alpha is not None else None, persistent=False)

    def forward(self, logits, targets):
        log_probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
        probabilities = log_probabilities.exp()

        log_pt = log_probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_term = (1 - pt) ** self.gamma
        loss = -focal_term * log_pt

        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class TwoStageFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, alpha_stage_1=None, alpha_stage_2=None, stage_2_weight=1.0, reduction='mean'):
        super().__init__()
        self.stage_1_loss = FocalLoss(gamma=gamma, alpha=alpha_stage_1, reduction=reduction)
        self.stage_2_loss = FocalLoss(gamma=gamma, alpha=alpha_stage_2, reduction=reduction)
        self.stage_2_weight = stage_2_weight

    def forward(self, contact_logits, class_logits, labels):
        contact_target = (labels != 0).long()
        loss_1 = self.stage_1_loss(contact_logits, contact_target)

        contact_mask = labels != 0
        if contact_mask.any():
            class_target = labels[contact_mask] - 1
            loss_2 = self.stage_2_loss(class_logits[contact_mask], class_target)
        else:
            loss_2 = contact_logits.new_tensor(0.)

        return loss_1 + self.stage_2_weight * loss_2


def predict_labels(contact_logits, class_logits):
    # "translation layer" for the fact that the models predicts in two stages
    with torch.no_grad():
        contact_pred = contact_logits.argmax(dim=1)
        class_pred = class_logits.argmax(dim=1) + 1
        return torch.where(contact_pred == 0, torch.zeros_like(class_pred), class_pred)


def train_model(model, training_dataloader, validation_dataloader, criterion, optimizer, scheduler, epochs, patience, device, f1_average='macro'):
    training_history = {'train_loss': [], 'val_loss': [], 'train_f1': [], 'val_f1': []}

    best_val_f1 = -float('inf')
    best_model_state = None

    early_stopping_counter = 0

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_loss, train_total = 0.0, 0
        train_predictions, train_labels = [], []

        for samples_batch_s, samples_batch_t, labels_batch in training_dataloader:
            samples_batch_s, samples_batch_t, labels_batch = samples_batch_s.to(device), samples_batch_t.to(device), labels_batch.to(device)

            optimizer.zero_grad()
            contact_logits, class_logits = model(samples_batch_s, samples_batch_t)
            loss = criterion(contact_logits, class_logits, labels_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(labels_batch)
            predicted = predict_labels(contact_logits, class_logits)

            train_predictions.append(predicted.detach().cpu())
            train_labels.append(labels_batch.detach().cpu())
            train_total += len(labels_batch)

        avg_train_loss = train_loss / train_total
        train_predictions = torch.cat(train_predictions).numpy()
        train_labels = torch.cat(train_labels).numpy()
        avg_train_f1 = f1_score(train_labels, train_predictions, average=f1_average, zero_division=0)

        # Validation
        model.eval()
        val_loss, val_total = 0.0, 0
        val_predictions, val_labels = [], []

        with torch.no_grad():
            for samples_batch_s, samples_batch_t, labels_batch in validation_dataloader:
                samples_batch_s, samples_batch_t, labels_batch = samples_batch_s.to(device), samples_batch_t.to(device), labels_batch.to(device)

                contact_logits, class_logits = model(samples_batch_s, samples_batch_t)
                loss = criterion(contact_logits, class_logits, labels_batch)

                val_loss += loss.item() * len(labels_batch)
                predicted = predict_labels(contact_logits, class_logits)

                val_predictions.append(predicted.detach().cpu())
                val_labels.append(labels_batch.detach().cpu())
                val_total += len(labels_batch)

            avg_val_loss = val_loss / val_total
            val_predictions = torch.cat(val_predictions).numpy()
            val_labels = torch.cat(val_labels).numpy()
            avg_val_f1 = f1_score(val_labels, val_predictions, average=f1_average, zero_division=0)

        # Scheduler
        scheduler.step()

        # Checkpoint + Early Stopping based on validation F1
        if avg_val_f1 > best_val_f1:
            best_val_f1 = avg_val_f1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        # Logging
        training_history['train_loss'].append(avg_train_loss)
        training_history['val_loss'].append(avg_val_loss)
        training_history['train_f1'].append(avg_train_f1)
        training_history['val_f1'].append(avg_val_f1)

        print(f"Epoch {epoch:3d}/{epochs} "
              f"| Train Loss: {avg_train_loss:.4f} ~  F1 Score: {avg_train_f1:.4f} "
              f"| Val Loss: {avg_val_loss:.4f} ~  F1 Score: {avg_val_f1:.4f}")

        # Early Stopping Check
        if early_stopping_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch} epochs ")
            break

    # Best model from val F1
    model.load_state_dict(best_model_state)

    return model, training_history


def save_model_in_directory(model, directory, label_encoder=None, feature_encoders=None, name_of_the_model_file="predictor_weights.pth", name_of_the_label_encoder_file="label_encoder.pkl", name_of_the_feature_encoders_file="feature_encoders.pkl"):
    directory_path = Path(directory)
    directory_path.mkdir(parents=True, exist_ok=True)

    model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    torch.save(model_state, directory_path / name_of_the_model_file)

    if label_encoder is not None:
        with open(directory_path / name_of_the_label_encoder_file, "wb") as f:
            pickle.dump(label_encoder, f)

    if feature_encoders is not None:
        with open(directory_path / name_of_the_feature_encoders_file, "wb") as f:
            pickle.dump(feature_encoders, f)