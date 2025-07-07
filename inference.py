import os

import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import torchvision.models as models
from torch.nn import functional as F

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from image_dataset import ImageDataset
from helper_functions import set_seed


# ################################################################################
#           ENVIRONMENT SET-UP
# ################################################################################

# Specify GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

print(torch.cuda.get_device_name(0))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Specify paths
repository_root = os.path.dirname(os.path.abspath(__file__))
results_directory = os.path.join(repository_root, 'final_model')
images_directory = os.path.join(repository_root, 'Data_examples')
inference_directory = os.path.join(repository_root, 'Inference_results')

# Specify other environment variables
SEED = 666
set_seed(SEED)


# ################################################################################
#           DATA PREPARATION
# ################################################################################

ZOOPLANKTON_CLASSES = os.listdir(images_directory)
NUM_CLASSES = len(ZOOPLANKTON_CLASSES)

# Get Classes from Orig Dataset
metadata_path = os.path.join(results_directory, 'environment.pth')
metadata = torch.load(metadata_path, weights_only = False)

classes, class_map = metadata['classes'], metadata['class_map']
class_map_rev = {v: k for k, v in class_map.items()}

inference_class_indices = [class_map[cls_name] for cls_name in ZOOPLANKTON_CLASSES]

# Define Inference Dataset
inference_dataset = ImageDataset(
    data_directory = images_directory,
    class_names = ZOOPLANKTON_CLASSES,
    class_indices = inference_class_indices,
    image_resolution = 64,
    image_transforms = None,
    seed = SEED
)

# Add Image Transformations to Inference Pipeline
inference_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(180),
    transforms.Pad(padding = 5, fill = 0),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
]) # same as train

inference_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
]) # basic

inference_dataset.append_image_transforms(
    image_transforms = inference_transforms, verbose = False
)

# Construct Inference Data Loader
inference_indices = list(range(len(inference_dataset)))

inference_loader = DataLoader(
    dataset = Subset(inference_dataset, inference_indices),
    batch_size = 1,
    shuffle = False
)


# ################################################################################
#           LOAD MODEL WITH WEIGHTS
# ################################################################################

model = models.densenet121(weights=None)
model.classifier = torch.nn.Linear(model.classifier.in_features, 14)

weights_path = os.path.join(results_directory, 'weights.pth')
weights = torch.load(weights_path, map_location = device)

model.load_state_dict(weights)
model.to(device)


# ################################################################################
#           INFERENCE
# ################################################################################

model.eval()
images, labels, probs, preds = [], [], [], []

with torch.no_grad():
    for image, label in inference_loader:
        image = image.to(device)
        label = label.to(device)
                
        output = model(image)
        prob = F.softmax(output, dim = 1)
        pred = output.argmax(dim = 1)

        images.append(image.cpu())
        labels.append(label)
        probs.append(prob)
        preds.append(pred)

images, labels, probs, preds = torch.cat(images), torch.cat(labels), torch.cat(probs), torch.cat(preds)


# ################################################################################
#           PLOT
# ################################################################################

max_images_per_row = 6
unique_classes = torch.unique(labels)

for cls_id in unique_classes:

    cls_indices = (labels == cls_id).nonzero(as_tuple = True)[0]
    
    # Set-up grid
    n_samples = len(cls_indices)
    n_cols = int(np.ceil(np.sqrt(n_samples)))
    n_rows = int(np.ceil(n_samples / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize = (n_cols * 2, n_cols * 2))
    axes = axes.flatten() if n_samples > 1 else [axes]

    # Plot each image
    for ax, idx in zip(axes, cls_indices):
        image = images[idx].mean(0).numpy()
        label = labels[idx].item()
        pred = preds[idx].item()

        ax.imshow(image, cmap = 'gray')
        ax.set_title(f'Pred: {class_map_rev[pred]}', fontsize = 7)

        colour = 'green' if label == pred else 'red'
        rect = patches.Rectangle(
            (0, 0), 1, 1, transform = ax.transAxes, linewidth = 4, edgecolor = colour, facecolor = 'none'
        )
        ax.add_patch(rect)
        ax.axis('off')

    # Set unused axes to blank
    for j in range(len(cls_indices), len(axes)):
        axes[j].axis('off')

    cls_name = class_map_rev[cls_id.item()]
    plt.suptitle(f'True Class: {cls_name}', fontsize=14)
    plt.tight_layout()

    # Save file
    inference_plot_path = os.path.join(inference_directory, f'{cls_name}.png')
    plt.savefig(inference_plot_path, dpi = 300, bbox_inches = 'tight')
    plt.close()

