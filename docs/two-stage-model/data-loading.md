# Data Loading

This section explains the modules that take the local data and put them in a form that is usable by the following code.

## `src.utils.input_preprocessing`

This module is the one that specifically manages the local files and makes them usable inside the code.\
The main functions of this module are two:

### `preprocess_data_files_from_path()`

This function takes as input a `path` (given in the `main.py` from the configuration), and a `feature_column` that represents the columns that will be extracted as features from the various `.tsv` training files.
The steps are the following:
1. Takes all the files in the directory path and combines them into a single `dataframe`.
2. If in any row of the `dataframe` there are any missing values in the features, the entire row gets deleted.
3. Finally, it tweaks the labels by replacing the empty values with `'Missing'`.

### `get_label_encoder_from_dataframe()`

This function takes the entire column that contains the labels from the `dataframe` that was given as an output from the previous function.\
The output of the function is a `LabelEncoder`; this aids the interpretability of the model's prediction.\
A particular feature in this function is that it forces the `'Missing'` label to be the first one; this is done because the input of the predictor model requires this particular label to be first, otherwise it won't be able to work correctly in a two-stage manner.


## `src.FeatureDataset`

This module defines one of the most important components in PyTorch projects, the `Dataset`.\
The class defined has the role of taking the preprocessed `dataframe` and transforming it in a structure that contains `torch.tensor` data directly usable by any later defined model.\
Three crucial functions are defined inside it:

### `__init__()`

This is the function that always runs when an instance of the class gets created. The main thing it does is take the input data and normalize it (if necessary).

One crucial variable that is asked in input is the boolean `fit`. This determines if the `Dataset` being created is a class that will be used in the training process (taking the labeled dataset), or it will be used to infer the predictions (using unlabeled data).\
If `fit=True`, the initialization will create encoders based on the data given as input, otherwise it will need already fitted encoders to transform the data into the encoded version. It will also save the labels, but only after being transformed using a `LabelEncoder` previously defined and given as input to this class.

The various inputs are managed in the following way:
- `'ss8', '3di_state'`: they represent a category, so for each of them an `OneHotEncoder` will be created to represent the specific state.
- `'phi', 'psi'`: angles in radians, not modified.
- *Remaining values*: the rest of the float values are normalized via a `StandardScaler`.

One last feature of this function is that it splits the data into two halves, one given to the source residue, the other for the target residue of the contact.
This is done because the model embeds two residues as input; therefore, it requires that the data is saved in two different variables. 

### `__len__()`

Simple function that returns the number of samples that are included in the dataset (corresponding to the number of rows in the preprocessed dataframe).

### `__getitem__()`

This function returns one sample given the index of the sample in the dataset.\
Since this class supports being used for training and inference, the `fit` variable is being used here.\
If `fit=True`, the function returns the data of the two residues and the label, otherwise it just gives the data. 

## `src.utils.dataset_utils`

This module defines two functions:

### `stratified_split()`

This function applies a stratified split from a defined `Dataset` following a split ratio (relative size of the validation set).\
The function returns two `Subset` of the dataset, one being the training set, the other being the validation set.

### `get_class_weights_from_dataframe()`

This function calculates the *alpha* values used by the `FocalLoss` function during the training process.\
Given all the training labels, it computes how the classes will be weighted based on how frequent those classes are in the dataset. 

## Further Reading

Next up in the pipeline is the [Model](model.md).