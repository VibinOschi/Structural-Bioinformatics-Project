# Model and Hyperparameters

This section of the documentation will detail another relevant part of this particular pipeline: the *Two-Stage Model*.\
This documentation page will mostly talk about the code of the model itself and some supporting hyperparameters that are needed to run the model. The actual training and evaluation (with inference procedure), is documented [here](train-val-infer-procedures.md).

## `src.Predictor`

This module defines the model that learns how to detect contacts between the features that are given as input.
There are two intertwined classes defined here:

### `ResidueEncoder`

This is a simple linear network that is used by the following class.
Its job is to take one of the two residues and encode it, as later it will be used as input for the `Predictor` class, that concatenates the two encoders.

### `Predictor`

This is the main class that has to be defined, and it's the one that gives the classification result of the contact.

The initialization defines several things:
- The encoders for the residues, that can be either shared (same for both), or not (two different ones). This is done via the `shared_encoder` boolean in the model's parameters.
- The linear layers that learn to predict based on the input encoders.
- Two prediction heads that enable the *Two-Stage* prediction of the model.

When doing forward passes (as seen in the `forward()` function), the model concatenates the two encoders and passes the information through the linear layers. Finally, the results are given by the two heads, which give out 2 and 7 output classes respectively.\
The first head is to determine if it's a contact or the `'Missing'` class, while the other one is to determine which class between the contacts it is.

## Hyperparameters

The main focus here is the hyperparameters found in the `train.py` script, but the most noteworthy one is located at the top of the `src.utils.training_utils` module.

### `TwoStageFocalLoss`

This is the loss function (or criterion) that enables the training of the model architecture explained previously.
It takes the `FocalLoss` class that is defined above it and applies it in two different stages of loss, with the second one having a hyperparameter, that allows one to tune how strongly it affects the final loss when calculating one of the 7 contacts.

The FocalLoss is used in the training procedure to address the class imbalance, as many of them are under-represented.

### Other Hyperparameters

The other factors that influence the training that are worth noting are the `Adam` optimizer and the `CosineAnnealing` scheduler. These are other components that make the training process happen.

All the hyperparameters that are tunable are inside `configuration.json`, apart from the hard-coded variables that should not be changed.

## Further Reading

Training, Evaluation and Inference procedures can be found in [this page](train-val-infer-procedures.md).