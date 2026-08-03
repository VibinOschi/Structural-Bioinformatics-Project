# Training

This pipeline, contrary to the other one, does everything inside a single script, with the exception of getting the training `dataframe`.

Other than the `main` function, only an additional function is defined in the same file, that being `build_feature_matrix()`, which emulates the `Dataset` from the other pipeline. It encodes and normalizes the features that need it.\
Encoding applies to `ss8` and `3di_state` features that represent a category, while most of the float values are normalized, apart the two radian angles `phi` and `psi`.

This pipeline also applies a solution for class imbalance by calculating the class weights, a variable that is used when fitting the model.

For `XGBoost`, it's best to check the official documentation.
The only operational note given here is that it's set up by default to run on CUDA-supporting GPUs. To change that, uncomment the only comment in the model definition and comment out the two lines below it to switch to CPU.\
For best performance when running on CPU, the `n_jobs` variable should be tuned to the number of performance cores that the CPU has, or set it to `-1` to utilize all cores.

After the training is completed, the model gets saved with the label and feature encoders that are needed to run the model in a different instance without breaking it.

It then computes the same evaluation steps as the [other pipeline](../two-stage-model/train-val-infer-procedures.md#validation-procedure).

# Inference

The inference procedure is not too complex.\
The script creates a new instance of the `XGBoost` model and loads it from local storage with the two other encoders that are needed to run it.

For every PDB file present in the configured input directory, it builds a feature matrix that is used to infer the output and the confidence score.

Finally, these two outputs are added as columns in the `dataframe` created from the PDBs, and only then, it gets saved in a local output directory.