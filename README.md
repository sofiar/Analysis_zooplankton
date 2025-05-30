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
 1. Create python environment 
 2. Install needed packages
