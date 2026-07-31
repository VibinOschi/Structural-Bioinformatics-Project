import torch
import pickle
import os

from src.Predictor import Predictor
from src.utils.training_utils import predict_labels

def load_predictor(config, device):
    model = Predictor(dropout=config['dropout'], shared_encoder=True).to(device)

    state_dict = torch.load(
        os.path.join(config['predictor_model_path'], 'predictor_weights.pth'),
        map_location=device
    )
    model.load_state_dict(state_dict)
    model.eval()

    label_encoder_path = os.path.join(config['predictor_model_path'], 'label_encoder.pkl')
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)

    feature_encoders_path = os.path.join(config['predictor_model_path'], 'feature_encoders.pkl')
    with open(feature_encoders_path, 'rb') as f:
        feature_encoders = pickle.load(f)

    return model, label_encoder, feature_encoders


def run_inference(model, dataloader, label_encoder, device):
    predicted_labels = []
    scores = []

    with torch.no_grad():
        for batch in dataloader:
            residue_features_s, residue_features_t = batch[0].to(device), batch[1].to(device)

            contact_logits, class_logits = model(residue_features_s, residue_features_t)

            contact_probabilities = torch.softmax(contact_logits, dim=1)
            class_probabilities = torch.softmax(class_logits, dim=1)

            predicted_indices = predict_labels(contact_logits, class_logits)

            # Confidence -> P(no contact) ['Missing']
            # otherwise -> P(contact) * P(predicted class | contact)
            is_contact = predicted_indices != 0
            class_confidence, _ = torch.max(class_probabilities, dim=1)
            confidence = torch.where(
                is_contact,
                contact_probabilities[:, 1] * class_confidence,
                contact_probabilities[:, 0],
            )

            predicted_labels.extend(label_encoder.inverse_transform(predicted_indices.cpu().numpy()))
            scores.extend(confidence.cpu().numpy().tolist())

    return predicted_labels, scores