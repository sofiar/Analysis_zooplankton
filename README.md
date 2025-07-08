# Analysis_zooplankton

This repository contains scripts to train a CNN to classify images of Zooplankton species and run inference.
The dataset was provided by the Ministry of Natural Resources and Forestry (MNR), Government of Ontario.

The model is trained to classify the following 14 species:
`Bosmina`, `Bubbles`, `Calanoid`, `Chydoridae`, `Chironomid`, `Cyclopoid`, `Daphnia`, `Floc 1`, `Floc 2`, `Harpacticoid`, `LargeZ-1`, `Nauploo`, `Sididae`, `TooSmall`

Note that in some analyses, the less important classes (Bubbles, Floc 1 & 2, LargeZ-1, Nauploo, Sididae, TooSmall) are considered as one class: Other.

## 📂 Contents
### 🗄️ Scripts
* `helper_functions.py`: Contains utility functions for environment set-up and minor data processing
* `image_dataset.py`: Custom PyTorch `Dataset` to load and process images for training.
* `inference.py`: Runs inference using trained model weights on images in **Data_examples/** and outputs results to **Inference_results/**.
* `train.py`: Trains final CNN based on results from development and tuning. Saves model weights and metadata to **final_model/**.
### 📚 Notebooks
* `visualize_results_merged.ipynb`: Notebook to explore test performance, using merged 9 classes.
* `visualize_results.ipynb`: Notebook to explore training metrics and test performance, using all 14 classes.
### 📊 Data & Directories
* **Data_examples/**: Contains `.tif` images of zooplankton used for inference.
* **final_model/**: Contains final model weights and model metadata.
* **Inference_results/**: Contains `.png` images of inference results by class and a `.txt` file with overall metrics.

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

## 🔁 Reproducible Inference Example
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

 2. Run the inference script.
 ```
 python inference.py
 ```

 3. Explore results in the **Inference_results/** directory. Each `.png` corresponds to 1 of 14 zooplankton classes listed above. The `.txt` file contains metrics for both the original 14 classes and the merged classes.
   ```plaintext
   Inference_results/
   ├── metrics.txt
   ├── Calanoid_1.png
   ├── Daphnia.png
   ├── ...
   ```

   




