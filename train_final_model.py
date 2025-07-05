import os
import torch
from torchvision import transforms

from helper_functions import set_seed

from image_dataset import ImageDataset
from model import Model

# ################################################################################
#           ENVIRONMENT SET-UP
# ################################################################################

# Specify GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Specify paths
data_directory = '/data/zooplankton_data'
results_directory = '/home/bushra/Analysis_zooplankton/'

# Specify other environment variables
SEED = 666
set_seed(SEED)


# ################################################################################
#           GLOBAL VARIABLES
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

train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(180),
    transforms.Pad(padding = 5, fill = 0),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])


# ################################################################################
#           PREPARE DATA
# ################################################################################

dataset = ImageDataset(
    data_directory = data_directory,
    class_names = ZOOPLANKTON_CLASSES,
    max_class_size = 15000,
    image_resolution = 64,
    image_transforms = None,
    seed = SEED
)

dataset.append_image_transforms(
    image_transforms = train_transforms, verbose = False
)

train_split, val_split, test_split = dataset.split_train_test_val(
    train_prop = 0.7, val_prop = 0.1, test_prop = 0.2, verbose = False
)

train_sample_weights, train_class_weights = dataset.compute_sample_weights(
    train_split, weights = 'softmax_inverse'
)

train_loader, val_loader, test_loader = dataset.create_dataloaders(
    batch_size = 64,
    train_indices = train_split,
    val_indices = val_split,
    test_indices = test_split,
    image_transforms = None,
    train_sample_weights = None
)


# ################################################################################
#           TRAIN FINAL MODEL
# ################################################################################

HYPERPARAMETERS = {
    'loss_fn': {'type': 'CrossEntropyLoss', 'weights': train_class_weights}, 
    'optimizer': 'Adam', 
    'lr': 5e-4, 
    'epochs': 80, 
    'scheduler':{'type': 'CosineAnnealingLR', 'T_max': 50},
    'early_stopping': {'patience': 15, 'delta': 0.005},
    'batch_size': 64
}

model = Model(
    data_directory = data_directory,
    device = device,
    num_classes = 14,
    model_name = 'densenet121'
)

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
run_name = f'{MODEL_ID}_final'

metadata = {
    'model_id': MODEL_ID,
    'model_name': 'densenet121',
    'run_name': run_name,
    'dataset': dataset,
    'classes': dataset.class_names,
    'class_map': {name: idx for name, idx in zip(dataset.class_names, dataset.class_indices)},
    'train_metrics': model.train_results,
    'test_loader': test_loader,
    'hyperparameters': HYPERPARAMETERS,
    'image_transforms': dataset.image_transforms,
    'max_class_size': 15000,
}

SAVE = True

if SAVE:
    print(f'Saving weights, predictions, and metadata.')
    print(f'Run Name: {run_name}')

    # Save learned weights, predictions and results
    torch.save(model.model.state_dict(), os.path.join(results_directory, 'weights', run_name + '.pth'))
    torch.save((labels, probs, preds), os.path.join(results_directory, 'predictions', run_name + '.pth'))
    torch.save(metadata, os.path.join(results_directory, 'environment', run_name + '.pth'))

# Delete model objects
del model

