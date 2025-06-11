import os
import copy

import torch
from torchvision import transforms
from torch.utils.data import random_split, Subset, DataLoader, SequentialSampler
import torchvision.models as models

from modular.engine import accuracy_fn
from helper_functions import set_seed

from image_dataset import ImageDataset
from model import Model

# ################################################################################
#           ENVIRONMENT SET-UP
# ################################################################################

# Specify GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

print(torch.cuda.get_device_name(0))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Specify paths
data_directory = '/data/zooplankton_data'
results_directory = '/home/bushra/Analysis_zooplankton/'

# Specify other environment variables
SEED = 666
set_seed(SEED)


# ################################################################################
#           LOAD & TRANSFORM DATA
# ################################################################################

MAX_CLASS_SIZE = 2000

IMAGE_RESOLUTION = 64
IMAGE_PADDING = 5
IMAGE_FILL = 0
IMAGE_ROTATION = 180

ZOOPLANKTON_CLASSES = ['Daphnia', 'Calanoid_1', 'Cyclopoid_1']
NUM_CLASSES = len(ZOOPLANKTON_CLASSES)

# Image Transformations
general_transforms = transforms.Compose([
    transforms.Pad(padding = IMAGE_PADDING, fill = IMAGE_FILL),
    transforms.Resize((IMAGE_RESOLUTION, IMAGE_RESOLUTION)),
    transforms.ToTensor()
])

train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(IMAGE_ROTATION),
    transforms.Pad(padding = IMAGE_PADDING, fill = IMAGE_FILL),
    transforms.Resize((IMAGE_RESOLUTION, IMAGE_RESOLUTION)),
])

# Define Dataset
dataset = ImageDataset(
    data_directory = data_directory,
    class_names = ZOOPLANKTON_CLASSES,
    max_class_size = MAX_CLASS_SIZE,
    image_resolution = IMAGE_RESOLUTION,
    image_transforms = general_transforms,
    seed = SEED
)

NUM_SAMPLES = len(dataset)

dataset.print_dataset_details()


# ################################################################################
#           TRAIN, TEST & VALIDATION
# ################################################################################

TRAIN_PROP = 0.7
VAL_PROP = 0.1
TEST_PROP = 0.2

BATCH_SIZE = 80

torch.manual_seed(SEED)

# Split Dataset
train_split, val_split, test_split = random_split(
    range(NUM_SAMPLES),
    lengths = [TRAIN_PROP, VAL_PROP, TEST_PROP],
    generator = torch.Generator().manual_seed(SEED)
)

print(f'\nTrain Samples: {len(train_split.indices)}')
print(f'Val Samples: {len(val_split.indices)}')
print(f'Test Samples: {len(test_split.indices)}')

# Transform All Samples
dataset_aug = copy.deepcopy(dataset)
dataset_aug.image_transforms = transforms.Compose(
    general_transforms.transforms + train_transforms.transforms
)

# Create Subsets
train_dataset = Subset(dataset_aug, train_split.indices)
val_dataset = Subset(dataset, val_split.indices)
test_dataset = Subset(dataset, test_split.indices)

# Data Loaders
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, 
    generator = torch.Generator().manual_seed(SEED)
)
val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = False, 
    generator = torch.Generator().manual_seed(SEED)
)
test_loader = DataLoader(
    test_dataset, batch_size = BATCH_SIZE, sampler = SequentialSampler(test_dataset)
)


# ################################################################################
#           HYPERPARAMETER TUNING
# ################################################################################

# Specify Model [UPDATE THIS]
MODEL_NAME = 'densenet121' # densenet121, resnet50

# Specify 
HYPERPARAMETER_SEARCH_GRID = {
    'loss_fn': ['CrossEntropyLoss'],
    'optimizer': ['Adam'],
    'lr': [1e-3],
    'epochs': [10],
    'scheduler': [{'type': 'StepLR', 'step_size': 10, 'gamma': 0.1}],
    'early_stopping': [{'patience': 10, 'delta': 0.005}]
}

model = Model(
    data_directory = data_directory,
    device = device,
    num_classes = NUM_CLASSES,
    model_name = MODEL_NAME
)

HYPERPARAMETERS, _ = model.gridsearch(
    parameter_grid = HYPERPARAMETER_SEARCH_GRID,
    train_loader = train_loader,
    val_loader = val_loader,
    scoring_fn = accuracy_fn
)


# ################################################################################
#           TRAIN MODEL & PREDICT
# ################################################################################

model.train(
    hyperparameters = HYPERPARAMETERS,
    train_loader = train_loader,
    val_loader = val_loader
)

labels, probs, preds = model.predict(test_loader = test_loader)


# ################################################################################
#           SAVE WEIGHTS, RESULTS & PREDICTIONS
# ################################################################################

MODEL_ID = model.model_id
run_name = f'{MODEL_ID}_{MODEL_NAME}.pth'

metadata = {
    'model_id': MODEL_ID,
    'model_name': MODEL_NAME,
    'dataset': dataset,
    'classes': dataset.class_names,
    'class_idx': {name: idx for name, idx in zip(dataset.class_names, dataset.class_idx)},
    'train_metrics': model.train_results,
    'test_loader': test_loader,
    'hyperparameters': HYPERPARAMETERS
}

# Save learned weights, predictions and results
torch.save(model.model.state_dict(), os.path.join(results_directory, 'weights', run_name))

torch.save((labels, probs, preds), os.path.join(results_directory, 'predictions', run_name))

torch.save(metadata, os.path.join(results_directory, 'environment', run_name))

# Delete model objects
del model

