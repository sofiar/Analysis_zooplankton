import os
import torch
from torchvision import transforms

from helper_functions import set_seed

from image_dataset import ImageDataset
import time
from datetime import datetime
from itertools import product

import torch
import torchvision.models as models
from torch.nn import functional as F
from modular import engine

from helper_functions import set_seed, extract_metrics


# ################################################################################
#           ENVIRONMENT SET-UP
# ################################################################################

# Specify GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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
#           SELECT CLASSES
# ################################################################################

ZOOPLANKTON_CLASSES = [
    'Bosmina_1',
    'Bubbles',
    'Calanoid_1',
    'Chironomid',
    'Chydoridae',
    'Cyclopoid_1',
    'Daphnia',
    'Floc_1',
    'Floc_2',
    'Herpacticoida',
    'LargeZ-1',
    'Nauplii',
    'Sididae',
    'TooSmall'
]
NUM_CLASSES = len(ZOOPLANKTON_CLASSES)


# ################################################################################
#           DATA PREPARATION
# ################################################################################

# Define Dataset
dataset = ImageDataset(
    data_directory = data_directory,
    class_names = ZOOPLANKTON_CLASSES,
    max_class_size = 15000,
    image_resolution = 64,
    image_transforms = None,
    seed = SEED
)

# Add Image Transformations to Pipeline
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(180),
    transforms.Pad(padding = 5, fill = 0),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

dataset.append_image_transforms(
    image_transforms = train_transforms,
    verbose = False
)

# Split Data into Train and Test
train_split, _, test_split = dataset.split_train_test_val(
    train_prop = 0.95, val_prop = 0, test_prop = 0.05, verbose = False
)

# Compute Weights per Class (using Train)
train_sample_weights, train_class_weights = dataset.compute_sample_weights(
    train_split, weights = 'softmax_inverse'
)

# Construct Data Loaders
train_loader, _, test_loader = dataset.create_dataloaders(
    batch_size = 64,
    train_indices = train_split,
    val_indices = [],
    test_indices = test_split,
    image_transforms = None,
    train_sample_weights = None
) # not weighted


# ################################################################################
#           MODEL DEFINITION & SET-UP
# ################################################################################

# Load Model & Weights
model = models.densenet121(weights = None)
weights_path = os.path.join(data_directory, 'densenet121-a639ec97.pth')

state_dict = torch.load(weights_path, map_location = 'cpu')
model.load_state_dict(state_dict, strict = False)
model.to(device)

# Set Final Layer
model.classifier = torch.nn.Linear(model.classifier.in_features, NUM_CLASSES)

# Model Set-Up
loss_fn = torch.nn.CrossEntropyLoss(weight = train_class_weights.to(device)) # with penalty
optimizer = torch.optim.Adam(params = model.parameters(), lr = 5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 50)
early_stopping = engine.EarlyStopping(patience = 15, delta = 5e-3)


# ################################################################################
#           TRAINING LOOP
# ################################################################################

print(f'\nStarting training!')

start = time.time()
train_results = engine.train_test_loop(
    model = model,
    train_dataloader = train_loader,
    test_dataloader = test_loader,
    optimizer = optimizer,
    loss_fn = loss_fn,
    epochs = 80,
    Scheduler = scheduler,
    early_stopping = early_stopping,
    device = device,
    print_b = True
)
elapsed = time.time() - start

print(f'\nTraining Finished! Time Elapsed: {elapsed:.2f} sec.')

train_results = extract_metrics(train_results)


# ################################################################################
#           TEST RESULTS
# ################################################################################

# Note there is data contamination because test was used for early stopping
# The model was properly validated during development without contamination

model.eval()
labels, probs, preds = [], [], []

with torch.no_grad():
    for image, label in test_loader:
        image = image.to(device)
        label = label.to(device)
                
        output = model(image)
        prob = F.softmax(output, dim = 1)
        pred = output.argmax(dim = 1)

        labels.append(label)
        probs.append(prob)
        preds.append(pred)

labels, probs, preds = torch.cat(labels), torch.cat(probs), torch.cat(preds)


# ################################################################################
#           SAVE MODEL WEIGHTS & METADATA
# ################################################################################

run_name = 'final_model'

metadata = {
    'model_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
    'model_name': 'densenet121',
    'run_name': run_name,
    'dataset': dataset,
    'classes': dataset.class_names,
    'class_map': {name: idx for name, idx in zip(dataset.class_names, dataset.class_indices)},
    'train_metrics': train_results,
    'test_loader': test_loader,
    'image_transforms': dataset.image_transforms,
    'max_class_size': 15000,
}

# Save learned weights, predictions and results
torch.save(model.state_dict(), os.path.join(results_directory, run_name, 'weights.pth'))
torch.save((labels, probs, preds), os.path.join(results_directory, run_name, 'predictions.pth'))
torch.save(metadata, os.path.join(results_directory, run_name, 'environment.pth'))

# Delete model objects
del model