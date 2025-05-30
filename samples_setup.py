################################################################################
################################# Samples Set up ###############################
################################################################################
import random
import os
import io
import tarfile
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset, Subset
import torch.nn.functional as F


class ImageDataset(Dataset):
    def __init__(self, folder_path,  transform = None, name_classes = None,
                 resolution: int = 28,num_files: int = 10000, n_files = None,
                 seed = 999):
        """
        Args:
            folder_path (str): Path to the dataset folder where subdirectories are class names.
            transform (callable, optional): Optional transformations to apply to the images.
            name_classes (callable, optional): Labels names.
            resolution: resolution to transform images.             
        """
        
        self.folder_path = folder_path
        self.transform = transform
        self.name_classes = name_classes
        self.resolution = resolution

        # Initialize image file paths and corresponding labels
        self.image_files = []
        self.labels = []
        
        if self.name_classes is None:
            name_classes = os.listdir(folder_path)
            
        if(n_files) is not None:
            dict_n_files = dict(zip(name_classes,n_files))
        else:
            dict_n_files = dict(zip(name_classes,[num_files]*len(name_classes)))
            

        # Load all image paths and their class names
        for class_name in name_classes:#os.listdir(folder_path):
            class_path = os.path.join(folder_path, class_name)
            if os.path.isdir(class_path):  # Only consider directories
                all_files = os.listdir(class_path)
                sample_size = min(num_files, len(all_files))
                dict_n_files[class_name] = sample_size
                # choose files randomly
                random.seed(seed)
                sample_files = random.sample(all_files, sample_size)
                for image_name in sample_files:
                    if image_name.lower().endswith('.tif'):  # Only load .tif files
                        self.image_files.append(os.path.join(class_path, image_name))
                        self.labels.append(class_name)
                     
        # Create a mapping of class names to integer indices
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(sorted(set(self.labels)))}
        self.labels = [self.class_to_idx[label] for label in self.labels]  # Convert labels to integers

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        image = Image.open(img_path)

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        # Get label
        label = self.labels[idx]

        # Convert label to a tensor
        label = torch.tensor(label, dtype=torch.long)

        return image, label
    
    
    
class ImageDatasetHierarchical(Dataset):
    def __init__(self, folder_path,  transform=None,name_classes=None,
                 resolution: int = 28,num_files: int = 10000,n_files=None):
        """
        Args:
            folder_path (str): Path to the dataset folder where subdirectories are class names.
            transform (callable, optional): Optional transformations to apply to the images.
            name_classes (callable, optional): Labels names.
            resolution: resolution to transform images.             
        """
        
        self.folder_path = folder_path
        self.transform = transform
        self.name_classes = name_classes
        self.resolution = resolution

        # Initialize image file paths and corresponding labels
        self.image_files = []
        self.labels = []
        self.main_labels = []

        if self.name_classes is None:
            name_classes = os.listdir(folder_path)
            
        if(n_files) is not None:
            dict_n_files = dict(zip(name_classes,n_files))
        else:
            dict_n_files = dict(zip(name_classes,[num_files]*len(name_classes)))
                

        # Load all image paths and their class names
        for class_name in name_classes:#os.listdir(folder_path):
            class_path = os.path.join(folder_path, class_name)
            if os.path.isdir(class_path):  # Only consider directories
                count = 0
                for image_name in os.listdir(class_path):
                    if image_name.lower().endswith('.tif'):  # Only load .tif files
                        self.image_files.append(os.path.join(class_path, image_name))
                        self.labels.append(class_name)
                        
                        if class_name == "Calanoid_1":
                            self.main_labels.append('Copepod')
                            
                        elif class_name == "Cyclopoid_1":
                            self.main_labels.append('Copepod')
                            
                        elif class_name == "Herpacticoida":
                            self.main_labels.append('Copepod')
                        
                        else: 
                            self.main_labels.append(class_name)
                                                    
                        count += 1
                        if count >= dict_n_files[class_name]:
                            break  # Stop after loading N files  

        # Create a mapping of class names to integer indices
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(sorted(set(self.labels)))}
        self.main_class_to_idx = {main_class_name: idx for idx, main_class_name in enumerate(sorted(set(self.main_labels)))}
        
        self.labels = [self.class_to_idx[label] for label in self.labels]  # Convert labels to integers
        self.main_labels = [self.main_class_to_idx[label] for label in self.main_labels]  # Convert main labels to integers

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        image = Image.open(img_path)

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        # Get label
        sub_label = self.labels[idx]
        main_label = self.main_labels[idx]

        # Convert label to a tensor
        sub_label = torch.tensor(sub_label, dtype=torch.long)
        main_label = torch.tensor(main_label, dtype=torch.long)


        return image,  main_label, sub_label
    
        
def transform_resize(resolution: int, pad: float = 5):
    """
    Create a resize transform pipeline 
    
    Args:
        resolution (int): Size to resize the images
        pad (float): Size for padding the images. Default = 5
    
    Returns:
        transforms.Compose: The transform pipeline.
    """    
    
    transformation = transforms.Compose([
        transforms.Pad(padding = pad, fill = 0),
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor()# Convert to PyTorch tensor
    ])
    
    return transformation


def transform_train(resolution: int,  pad: float = 5):
    """
    Create a data augmentation transformation including 
    RandomHorizontal, RandomVertical, RandomRotation, Pa
    and resize.
    
    Args:
        resolution (int): Size to resize the images
        pad (float): Size for padding the images. Default = 5
    
    Returns:
        transforms.Compose: The transform pipeline.
    """    
    
    transformation = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180),
        transforms.ToTensor(),
        transforms.Pad(padding = 5, fill = 0),
        transforms.Resize((resolution, resolution)),
    ])
    
    return transformation


def filter_by_class(dataset, target_class):
    """
    Filter a dataset to get a specific class.
    
    Args:
        dataset: The dataset to filter.
        target_class: The class number to filter for.
    
    Returns:
        A subset of the dataset containing only the target class.
    """
    labels = dataset.labels
    indices = torch.where(torch.tensor(labels) == target_class)[0].tolist()
    return Subset(dataset, indices)


def subsample_multiple_classes(dataset, class_samples, seed=None):
    """
    Subsample multiple classes and combine them into a single dataset.

    Args:
        dataset (Dataset): The original dataset.
        class_samples (dict): Dictionary where keys are class labels and values are the number of samples to draw.
        seed (int, optional): Random seed for reproducibility.
    
    Returns:
        Subset: A new dataset containing the combined subsamples.
    """
    if seed is not None:
        random.seed(seed)
    
    # Collect indices for each class
    combined_indices = []
    labels = dataset.labels

    for target_class, sample_size in class_samples.items():
        # Filter indices for the current class
        class_indices = torch.where(torch.tensor(labels) == target_class)[0].tolist()
                
        # Subsample the class indices
        sampled_indices = random.sample(class_indices, min(sample_size, len(class_indices)))
        
        # Add to the combined indices
        combined_indices.extend(sampled_indices)
    
    # Return a combined Subset
    return Subset(dataset, combined_indices)

def get_predictions(model, image, label,probs= False):
    # model.to(device)  # Ensure model is on GPU
    # image = image.to(device)  # Move image tensor to GPU
    # label = label.to(device)  # Move label tensor to GPU

    with torch.no_grad():  
        out = model(image)  # Forward pass
        if probs:
            probas = F.softmax(out, dim=1)  # Convert logits to probabilities
        y_pred = out.argmax(dim=1)  # Get predicted class indices (still a tensor)
        
    if probs:
        return label, y_pred, probas  
    else:
        return label, y_pred 
