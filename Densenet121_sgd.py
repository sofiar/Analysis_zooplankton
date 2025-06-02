################################################################################
#   script to train and test Desnet121 with SGD optimizer to zooplankton data  #
################################################################################

import torch
import numpy as np
from modular import engine
from torch.utils.data import DataLoader, random_split, SequentialSampler
from joblib import Parallel, delayed
import torchvision.models as models
import time
import os
import samples_setup

# Set which GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Set device
print(torch.cuda.get_device_name(0)) 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set paths 
DataPath = '/data/zooplankton_data'
print('Data file path: ', DataPath)

ResultsPath = '/home/ruizsuar/Analysis_zooplankton/Environments'
print('Results file path: ', ResultsPath)

PredictionsPath = '/home/ruizsuar/Analysis_zooplankton/Predictions'
print('Results file path: ', PredictionsPath)

MyWeightsPath = '/home/ruizsuar/Analysis_zooplankton/Weights'
print('Results file path: ', PredictionsPath)

# Define classes to include in the model
# all_classes = ['Bosmina_1','Bubbles','Calanoid_1','Chironomid','Chydoridae',
#                'Cyclopoid_1','Daphnia','Floc_1','Herpacticoida',
#                'LargeZ-1','Nauplii','TooSmall']
all_classes = ['Daphnia','Calanoid_1','Cyclopoid_1']


# ################################################################################
# ################################# Get data #####################################
# ################################################################################

RESOLUTION = 64
MAXVALUE = 2000

transform_resize = samples_setup.transform_resize(resolution = RESOLUTION)

all_datasets = samples_setup.ImageDataset(
    DataPath, 
    transform = transform_resize, 
    name_classes = all_classes,
    resolution = RESOLUTION,
    num_files = MAXVALUE,
    seed = 666
    )

classes_keys = list(all_datasets.class_to_idx.keys())

print(f"Length of dataset: {len(all_datasets)}")
print(f"Class-to-Index Mapping: {all_datasets.class_to_idx}")

name_classes = list(all_datasets.class_to_idx.keys())
length_classes = []
for cl in name_classes:
    num_class = all_datasets.class_to_idx[cl]
    only_class = samples_setup.filter_by_class(all_datasets, num_class)
    length_classes.append(len(only_class))
    print(f"Samples of {cl}: {len(only_class)}")

# ################################################################################
# ################## Define train test and validation sets #######################
# ################################################################################

BATCH_SIZE = 80

train_size = int(0.7 * len(all_datasets))
val_size = int(0.10 * len(all_datasets))
test_size = len(all_datasets) - train_size - val_size

print(f'train size: {train_size}')
print(f'test size: {test_size}')

torch.manual_seed(6)

# Split the combined dataset
train_dataset, val_dataset, test_dataset = random_split(
    all_datasets, 
    [train_size, val_size, test_size]
    )

# Extract the labels of the train subset
indices_train = train_dataset.indices  # Get train indices
train_labels = [all_datasets[i][1].item() for i in indices_train]  
  
# add augmentation to the train set
transform_train = samples_setup.transform_train(resolution= RESOLUTION)
train_dataset.dataset.transform = transform_train

train_loader = DataLoader(train_dataset , batch_size = BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE,
                         sampler = SequentialSampler(test_dataset))
val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle=True)

# ################################################################################
# ###################### Densenet121 - SGD optimizer ############################
# ################################################################################

EPOCHS = 40
                         
model_dn2 = models.densenet121(weights=None)
model_keys = set(model_dn2.state_dict().keys())

# load weights
weights_path =  DataPath +  "/densenet121-a639ec97.pth"
# send weights to gpu
state_dict =  torch.load(weights_path,map_location='cpu')
weight_keys = set(state_dict.keys())

# load weights into model
model_dn2.load_state_dict(state_dict,strict =False)
model_dn2.to(device)

model_dn2.classifier = torch.nn.Linear(model_dn2.classifier.in_features, 
                                      len(all_classes))

loss_fn = torch.nn.CrossEntropyLoss()
    
optimizer = torch.optim.SGD(params=model_dn2.parameters(), lr=1e-3) 
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

print("Starting Densenet121 -SGD Inference!")
early_stop = engine.EarlyStopping(patience=10, delta=0.005)

start = time.time()
output_dn2 = engine.train_test_loop(
    model_dn2,train_loader,
    val_loader, optimizer, loss_fn,
    epochs = EPOCHS, print_b = True,
    early_stopping = early_stop,
    Scheduler = scheduler,
    device = device)

end = time.time()
elapsed = end - start
print(f"It took: {elapsed} secs to run")

################################################################################
################## Save variables, weights and predictions #####################
################################################################################

# Save some important variables 
variables_to_save = {
    'EPOCHS': EPOCHS,
    'test_loader': test_loader,
    'all_classes': all_classes,
    'dataset':all_datasets,
    'classes_keys': classes_keys,
    'elapsed_time': elapsed,
    #'train_loader': train_loader,
    #'val_loader': val_loader   
    
}
where_to_save = ResultsPath +'/Env_result_Densenet121_sgd.pth'
torch.save(variables_to_save, where_to_save)
print('Environrment saved in: '+ where_to_save)  

# Save predicted labels 
model_dn2.eval() 
outputs = Parallel(n_jobs=10)(delayed(samples_setup.get_predictions)(
    model=model_dn2,
    image=imag.to(device),
    label=target
    ) for imag, target in test_loader)

true_labels = []
predict_labels = []

for true, pred in outputs:
    true_labels.append(true)
    predict_labels.append(pred)
    
true_labels = torch.cat(true_labels)
predict_labels = torch.cat(predict_labels)    

where_to_save = PredictionsPath + '/Pred_result_Densenet121_sgd.pth'
torch.save((true_labels, predict_labels), where_to_save)
print('Predictions saved in ', where_to_save)

# Save models weights 
where_to_save = MyWeightsPath  + '/Weights_Densenet121_sgd.pth'
torch.save(model_dn2.state_dict(), where_to_save)
print('Weights saved in ', where_to_save)
  
# Remove model object from memory
del model_dn2  
del output_dn2
