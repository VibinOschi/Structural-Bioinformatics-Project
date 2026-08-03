# Structural-Bioinformatics-Project

## Usage

Running the code doesn't require any additions of flags in the commands, almost all variables are handled in the `configuration.json`.\
As such, prior to running, checking the paths of the various directories needed to run the code is advised.

### Model Training

There are two models that can be trained with two different commands.

The better performing *XGBoost* model can be trained using the following command: 

`python train_xgboost.py`

While the other model uses the more general command:

`python train.py`

### Using the model for predictions

For the prediction of the contacts, a single script that unifies both models is used.
To choose which model to use, the script provides a single flag:

```shell
# XGBoost (default)
python predict.py

# Two-Stage Model
python predict.py -xgboost False 
```

### Requirements

The use of a virtual environment with the necessary packages is recommended. To aid installation, the `requirements.txt` file has been provided, and it can be used with the following command:

```shell
pip install -r requirements.txt
```

## Further documentation

A deeper dive on the code can be found in the `docs` directory, or by simply clicking [here](docs/overview.md).