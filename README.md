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
* `train.py`: Trains final CNN based on results from development and tuning. Saves model weights and metadata to **final_model/**.
### 📚 Notebooks
* `inference.ipynb`: Notebook to run inference on a single user-supplied image. For new, unlabelled images. Mimics use-case of model.
* `inference_examples.ipynb`: Runs inference using trained model weights on images in **Data_examples/**. For labelled data not included in train or test for the final model.
* `visualize_train_results.ipynb`: Notebook to explore training metrics and test performance. Can use the `merge` variable to toggle between original 14 classes and merged 7 classes.
### 📊 Data & Directories
* **Data_examples/**: Contains labelled `.tif` images of zooplankton used for inference.
* **final_model/**: Contains final model weights and model metadata.

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

## 🔁 Reproducible Inference Example (for Labelled Data)
 1. Ensure the following structure within the **Data_examples/** directory, where the name of each sub-directory MUST match 1 of 14 zooplankton classes listed above.
   ```plaintext
   Data_examples/
   ├── Calanoid_1/
   │   ├── calanoid_001.tif
   │   ├── calanoid_002.tif
   │   ├── ...
   ├── Daphnia/
   │   ├── daphnia_001.tif
   │   ├── daphnia_002.tif
   │   ├── ...
   ```
 2. Run all cells in `inference_examples.ipynb`.
 3. View plots at the end of the notebook. Each plot corresponds to the true class, while the subtitle above each image corresponds to the predicted class and the class probability.


## 🔁 Reproducible Inference Example (for New, Unabelled Image)
1. Change the `image_path` variable in the first code cell of the `inference.ipynb` notebook.
2. Run all cells.
3. View the image, the predicted class and the corresponding probability are displayed at the end of the notebook.
   




