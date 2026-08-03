# Procedures

## Training Procedure

This procedure takes place in the `train.py` script. The goal is to take a labeled dataset and train the model to predict as well as possible the relation between the features and the labels.

The main function in the `src.utils.training_utils` module is `train_model()`.\
This function takes a model, the train/val dataloaders and many hyperparameters, and goes through several cycles of training (limited by the number of `epochs`).

An epoch is divided into two macro steps: training and validation.\
They both base their indicative performance on the *F1 Score*. This score is also central to the early stopping implemented in the function. If many epochs in a row perform worse (based on the validation set), then the training stops early to avoid consuming resources while not producing any meaningful training improvements.\
The difference between the two macro steps is the set that they use (training/validation dataloaders), and the fact that the training step changes the weights of the model, while the validation step checks the performance of the model on data that the training doesn't use.

As the model predicts in two stages, the use of the function `predict_labels()` is required, otherwise predictions wouldn't be counted correctly.

A `training_history` variable is implemented to track the history of the loss and performance for both the training and validation steps.

---

It's also worth noting that in the same module, at the end, a function to save the model locally is defined (`save_model_in_directory()`).\
This saves the model and the feature and label encoders that are needed to correctly reconstruct the model with the same capabilities.

## Validation Procedure

This procedure takes place in the `train.py` script. The goal is to give a better evaluation of the performance of the model compared to the training procedure.

This procedure is similar to the one that takes place inside the training procedure, but the outputs are not the same.
This is what this procedure outputs:
- Classification Report: prints a text summary of the precision, recall, F1 score for each class.
- Loss & F1 Score graph through the various epochs.
- Confusion Matrix graph.
- Matthews Correlation Coefficient: measure of the quality of multiclass classifications.
- ROC Area-Under-the-Curve (text form).
- ROC Curve graph.

Every metric above is implemented via the `sklearn.metrics` package, with the exception for the training history across epochs.

Every graph is saved in the directory configured in the `configuration.json` file.

## Inference Procedure

This procedure takes place in the `predict.py` script, only if the flag is set to use the model from this *Two-Stage* pipeline.

This pipeline uses the `src.utils.inference_utils` module, which specifies two functions:

- `load_predictor()`: loads what has been saved with the last function mentioned in the [training procedure section](#training-procedure). Given the config and the device that will run the inference, it returns the model and the various encoders.
- `run_inference()`: it's similar to an epoch cycle of training (or validation), but other than getting the predictions, it also computes the confidence of said predictions. These two are the outputs that will be added to the final result of the whole procedure.

The general steps of the inference procedure are:
1. Load the dataset in `fit=False`. Then put into a `DataLoader`.
2. Inference is run with the `run_inference()` function call.
3. The predictions and confidence scores are added as new columns in the dataframe of the protein whose contact types are being predicted.

This step is repeated for each *PDB* file inside the configured directory, returning the results in a `.tsv` in another configured directory. 