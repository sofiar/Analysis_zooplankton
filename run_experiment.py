import os
import random
import time
from datetime import datetime

import torch
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import random_split, DataLoader, SequentialSampler

import torchvision.models as models
from modular import engine

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
    transforms.ToTensor(),
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
train_dataset, val_dataset, test_dataset = random_split(dataset,
    lengths = [TRAIN_PROP, VAL_PROP, TEST_PROP],
    generator = torch.Generator().manual_seed(SEED)
    )

print(f'Train Samples: {len(train_dataset.indices)}')
print(f'Val Samples: {len(val_dataset.indices)}')
print(f'Test Samples: {len(test_dataset.indices)}')

# Transform Train
train_dataset.dataset.transform = train_transforms # this automatically applies it to all subsets

# Data Loaders
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE,
                         sampler = SequentialSampler(test_dataset)
                        )

# ################################################################################
#           TRAIN MODEL & PREDICT
# ################################################################################

# Specify Model [UPDATE THIS]
MODEL_NAME = 'resnet50' # densenet121, resnet50

# Specify training & hyperparameters [UPDATE THIS]
HYPERPARAMETERS = {
    'loss_fn': 'CrossEntropyLoss',
    'optimizer': 'Adam',
    'lr': 1e-3,
    'scheduler': {
        'type': 'StepLR',
        'step_size': 10,
        'gamma': 0.1
    },
    'early_stopping': {
        'patience': 10,
        'delta': 0.005
    },
    'epochs': 3,
    'batch_size': BATCH_SIZE,
}

model = Model(
    data_directory = data_directory,
    device = device,
    num_classes = NUM_CLASSES,
    model_name = MODEL_NAME
)

model.train(
    train_loader = train_loader,
    val_loader = val_loader,
    hyperparameters = HYPERPARAMETERS
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
    'test_loader': test_loader,
    'hyperparameters': HYPERPARAMETERS
}

# # Save learned weights, predictions and results
# torch.save(model.model.state_dict(), os.path.join(results_directory, 'weights', run_name))

# torch.save((labels, probs, preds), os.path.join(results_directory, 'predictions', run_name))

# torch.save(metadata, os.path.join(results_directory, 'environment', run_name))

# Delete model objects
del model

