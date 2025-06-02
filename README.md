# Analysis_zooplankton

This repository contains scripts to train and test CNN models to classify zooplankton species.
The dataset was provided by the Ministry of Natural Resources and Forestry(MNR), Government of Ontario.

## 📂 Contents
### 🗄️ Scripts
* `samples_setup.py`: Contains functions and classes for preparing the data used for the analysis. 
* `Densenet121_adam.py`: Implements a training loop for a Densenet121 architecture with adam optimizer.
* `Densenet121_sgd.py`: Implements a training loop for a Densenet121 architecture with SGD optimizer.
* `Resnet50_adam.py`:  Implements a training loop for a Resnet50 architecture with adam optimizer.
* `Resnet50_sgd.py`:  Implements a training loop for a Resnet50 architecture with SGD optimizer.
### 📚 Notebooks
* `Explore_results.ipynb`: Jupyter notebook for loading, inspecting and visualizing model performance.
### 📊 Data
* **Data_examples/**: It contains some images examples for the classes *Calanoid* and *Daphnia*.

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
   * `sklearn`
   




