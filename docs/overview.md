# Overview

This project mainly uses the PyTorch framework to manage the dataset and to train and run the models.
The code can be seen in two different pipelines: *Two-Stage Model* and *XGBoost Model*.

The reason for the existence of two pipelines is explained in the report; what you need to know here is that the latter performs better, but it uses as dependencies some code from the former pipeline.
For this reason, the less-performing pipeline is explained first.

## Two-Stage Model Pipeline

The *Two-Stage Model* is made up of customized PyTorch modules, spanning from the `Dataset` to the training and the evaluation of the custom-defined model.

The training and inference of this specific architecture can be checked in the dedicated directory starting with the [loading of the data](two-stage-model/data-loading.md).

If you're only interested in one specific model, [here](two-stage-model/data-loading.md#srcutilsinput_preprocessing) is the only module that the XGBoost pipeline uses.

## XGBoost Model Pipeline

This pipeline uses simpler code, thanks to the implementation of a ready-to-use Gradient Boosting framework: *XGBoost*.
Because of this, no custom model needed to be defined, and the only code that was reused from the previous pipeline is from the `src.utils.input_preprocessing` module.

All the code has been put in a single, relatively small script, which is documented [here](xgboost/xgboost-pipeline-docs.md).