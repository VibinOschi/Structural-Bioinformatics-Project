import torch
import pickle

from pathlib import Path


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        self.register_buffer('alpha', alpha if alpha is not None else None, persistent=False)

    def forward(self, logits, targets):
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

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




def train_model(model, training_dataloader, validation_dataloader, criterion, optimizer, scheduler, epochs, patience, device):
    training_history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    best_val_loss = float('inf')
    best_model_state = None

    early_stopping_counter = 0

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for samples_batch_s, samples_batch_t, labels_batch in training_dataloader:
            samples_batch_s, samples_batch_t, labels_batch = samples_batch_s.to(device), samples_batch_t.to(device), labels_batch.to(device)

            optimizer.zero_grad()
            outputs = model(samples_batch_s, samples_batch_t)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(labels_batch)
            predicted = outputs.argmax(dim=1)
            train_correct += (predicted == labels_batch).sum().item()
            train_total += len(labels_batch)

        avg_train_loss = train_loss / train_total
        avg_train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for samples_batch_s, samples_batch_t, labels_batch in validation_dataloader:
                samples_batch_s, samples_batch_t, labels_batch = samples_batch_s.to(device), samples_batch_t.to(device), labels_batch.to(device)

                outputs = model(samples_batch_s, samples_batch_t)
                loss = criterion(outputs, labels_batch)

                val_loss += loss.item() * len(labels_batch)
                predicted = outputs.argmax(dim=1)
                val_correct += (predicted == labels_batch).sum().item()
                val_total += len(labels_batch)

            avg_val_loss = val_loss / val_total
            avg_val_acc = val_correct / val_total

        # Scheduler + Checkpoint
        scheduler.step()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        # Logging
        training_history['train_loss'].append(avg_train_loss)
        training_history['val_loss'].append(avg_val_loss)
        training_history['train_acc'].append(avg_train_acc)
        training_history['val_acc'].append(avg_val_acc)

        print(f"Epoch {epoch:3d}/{epochs} "
              f"| train loss {avg_train_loss:.4f}  acc {avg_train_acc:.4f} "
              f"| val loss {avg_val_loss:.4f}  acc {avg_val_acc:.4f}")

        # Early Stopping Check
        if early_stopping_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch} epochs ")
            break

    # Best model from val loss
    model.load_state_dict(best_model_state)

    return model, training_history


'''
def save_model_in_directory(model, directory, name_of_the_model_file="predictor_weights.pth"):
    directory_path = Path(directory)
    directory_path.mkdir(parents=True, exist_ok=True)

    model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    torch.save(model_state, directory_path / name_of_the_model_file)
'''


def save_model_in_directory(model, directory, label_encoder=None, feature_encoders=None,
                             name_of_the_model_file="predictor_weights.pth",
                             name_of_the_label_encoder_file="label_encoder.pkl",
                             name_of_the_feature_encoders_file="feature_encoders.pkl"):
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

# Note: the following code is how you load the model that was saved by the previous function
#
#   model = Predictor(args)
#   state_dict = torch.load("model/predictor_weights.pt", map_location=device)
#   model.load_state_dict(state_dict)
#   model.to(device)
# Additionally, for inference:
#   model.eval()