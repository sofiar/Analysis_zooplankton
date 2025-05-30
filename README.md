# Analysis_zooplankton

This repository contains scripts to train and test CNN models to classify zooplankton species.
The dataset was provided by the Ministry of Natural Resources and Forestry(MNR), Government of Ontario.

* samples_setup.py: defines functions and classes to prepare data for the analysis. 
* Densenet121_adam.py: train and test Densenet121 architecture with adam optimizer.
* Densenet121_sgd.py: train and test Densenet121 architecture with SGD optimizer.
* Resnet50_adam.py: train and test Resnet50 architecture with adam optimizer.
* Resnet50_sgd.py: train and test Resnet50 architecture with sgd optimizer.
* **/Data_examples**: It contains some images examples for the classes *Calanoid*, *Cyclopoid* and *Daphnia*. 

## Set up
 1. Load python in your environment
```
module load python
```

 2. Install InformedML-CV by : 
```
pip install git+https://github.com/sofiar/InformedML-CV.git
```
3. Install any dependencies needed: There may be some libraries that need to be installed or updated:
check if *joblib*, *torch* are installed



