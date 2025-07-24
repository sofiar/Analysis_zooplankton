# Analysis_zooplankton

This repository contains scripts to train a CNN to classify images of Zooplankton species and notebooks to run inference.
The dataset was provided by the Ministry of Natural Resources and Forestry (MNR), Government of Ontario.

The model is trained to classify the following 14 species:

`Bosmina`, `Bubbles`, `Calanoid_1`, `Chydoridae`, `Chironomid`, `Cyclopoid_1`, `Daphnia`, `Floc_1`, `Floc_2`, `Harpacticoid`, `LargeZ-1`, `Nauploo`, `Sididae`, `TooSmall`

Note that in some analyses, the less important classes (`Bubbles`, `Floc 1`, `Floc 2`, `LargeZ-1`, `Nauploo`, `Sididae`, `TooSmall`) are considered as one class: `Other`.

## 📂 Contents
### 🗄️ Scripts
* `helper_functions.py`: Contains utility functions for environment set-up and minor data processing.
* `image_dataset.py`: Custom PyTorch `Dataset` to load and process images for training.
* `model.py`: Custom class definition to train and run inference using DenseNet121 or ResNet50 models. Can run gridsearch for hyperparameter tuning.
* `run_experiment.py`: Trains one or multiple CNNs (i.e., gridsearch) and saves trained model weights and related metadata.
### 📚 Notebooks
* `calibration_results.ipynb`: Notebook to check model calibration and scaling results.
* `visualize_train_results.ipynb`: Notebook to explore training metrics and test performance. Can use the `merge` variable to toggle between original 14 classes and merged 7 classes.
### 📊 Data & Directories
* **Data_examples/**: Contains labelled `.tif` images of zooplankton used for inference.

Note that model metadata, test predicitions, and weights are saved to the **environment/**, **predictions/**, and **weights/** directories but those are only contained in the local repository.

## ⚙️ Set up
 1. If working on cluster or module system load python in your environment
```
module load python
```

 2. Install `InformedML-CV` by : 
```
pip install git+https://github.com/sofiar/InformedML-CV.git
```

3. Install any dependencies: Make sure required libraries are installed, including:
   * `torch`
   * `joblib`
   * `seaborn`
   * `matplotlib`
   * `sklearn`
   * `numpy`

## 🔁 Reproducible Experiment Example
1. Ensure **environment/**, **predictions/**, and **weights/** directories exist within your repository.
2. If you have additional training data, in the `run_experiment.py` script, change the `data_subdirectories` list variable to include any new data. Note that each new subdirectory should have the same file structure as the main `data_directory` where the name of each folder within the subdirectory MUST match 1 of 14 zooplankton classes listed above.
3. Change none, some, or all of the following parameters in the `run_experiment.py` script:
```
BATCH_SIZE = 64
MODEL_NAME = 'densenet121'
TUNE = False
HYPERPARAMETER_SEARCH_GRID = {...} # to gridsearch new params
HYPERPARAMETERS = {...} # to test new param values
```
4. You can also change the weights computation and where to apply weights within `run_experiment.py`:
   - See the `compute_sample_weights()` methods in `image_dataset.py` for more options for weight computations:
```
train_sample_weights, train_class_weights = dataset.compute_sample_weights(
    train_split, weights = 'softmax_inverse'
)
```

   - Sample dataset based on specified weights in `train_sample_weights`:
```
train_loader, val_loader, test_loader = dataset.create_dataloaders(
    batch_size = BATCH_SIZE,
    train_indices = train_split,
    val_indices = val_split,
    test_indices = test_split,
    image_transforms = None,
    train_sample_weights = None OR train_sample_weights
)
```

  - Apply a weighted penalty in the loss function by specifying class weights:
```
'loss_fn': {'type': 'CrossEntropyLoss', 'weights': None OR train_class_weights}, 
```

5. Run `run_experiment.py`. Make note of the model ID that is printed at the end (e.g., `20250717_184244_densenet121').
6. Change the value of `run_name` to the model ID in both the `calibration_results.ipynb` and `visualize_train_results.ipynb` notebooks. Run all cells and explore results.



