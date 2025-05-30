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
                        count += 1
                        if count >= dict_n_files[class_name]:
                            break  # Stop after loading N files  

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
    

class ImageDatasetFree(Dataset):
    def __init__(self, folder_path, name_files, transform=None,
                 resolution: int = 28):
        """
        Args:
            folder_path (str): Path to the dataset folder where subdirectories are class names.
            transform (callable, optional): Optional transformations to apply to the images.
            resolution: resolution to transform images.             
        """
        
        self.folder_path = folder_path
        self.transform = transform
        self.resolution = resolution

        # Initialize image file paths and corresponding labels
        self.image_files = []
        self.labels = []
        
        # Load all image paths and their class names
        for file in name_files:
            image_path = os.path.join(folder_path, file)
            class_name = file.split('/')[0]
            self.image_files.append(image_path)
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

class ImageSoftDataset(Dataset):
    def __init__(self, folder_path, transform=None,name_classes=None,
                 resolution: int = 28,num_files: int = 10000):
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
            
        # Remove CopepodSpp from Classes dictionary
        self.class_to_idx = {
            class_name: idx for idx, class_name in enumerate(self.name_classes) if class_name != 'CopepodSpp'
            }    

        # Load all image paths and their class names
        for class_name in name_classes:#os.listdir(folder_path):
            class_path = os.path.join(folder_path, class_name)
            if os.path.isdir(class_path):  # Only consider directories
                count = 0
                for image_name in os.listdir(class_path):
                    if image_name.lower().endswith('.tif'):  # Only load .tif files
                        self.image_files.append(os.path.join(class_path, image_name))
                        # Default one-hot encoding
                        label_vector = np.zeros(len(self.name_classes), dtype=np.float32)
                        
                        if class_name == "CopepodSpp":
                            # Assign 1/3 probability to class Calanoid, cyclopoid and Herpacticoida
                            label_vector[self.class_to_idx["Calanoid_1"]] = 1/3
                            label_vector[self.class_to_idx["Cyclopoid_1"]] = 1/3
                            label_vector[self.class_to_idx["Herpacticoida"]] = 1/3
                            
                        else:
                            # Hard label (probability 1)
                            label_vector[self.class_to_idx[class_name]] = 1.0
                        self.labels.append(label_vector)
                                          
                        #self.labels.append(class_name)
                        count += 1
                        if count >= num_files:
                            break  # Stop after loading N files  
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        image = Image.open(img_path)

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

         # Convert label to a tensor
        soft_label = torch.tensor(self.labels[idx], dtype=torch.float32)  # Soft label (probability distribution)

        return image, soft_label
    
class ImageDatasetSoftFree(Dataset):
    def __init__(self, folder_path,dict_classes, name_files, transform=None,
                 resolution: int = 28):
        """
        Args:
            folder_path (str): Path to the dataset folder where subdirectories are class names.
            transform (callable, optional): Optional transformations to apply to the images.
            resolution: resolution to transform images.             
        """
        
        self.folder_path = folder_path
        self.transform = transform
        self.resolution = resolution
        
        # Remove CopepodSpp from Classes dictionary
        self.class_to_idx = dict_classes

        class_names = list(dict_classes.keys())
        self.name_classes = class_names

        # Initialize image file paths and corresponding labels
        self.image_files = []
        self.labels = []
        
        # Load all image paths and their class names
        for file in name_files:
            image_path = os.path.join(folder_path, file)
            category_name = file.split('/')[0]
            self.image_files.append(image_path)
            # Default one-hot encoding
            label_vector = np.zeros(len(self.name_classes)+1, dtype=np.float32) # To account for CopepodSpp
            
            if category_name == "CopepodSpp":
                # Assign 1/3 probability to class Calanoid, cyclopoid and Herpacticoida
                label_vector[self.class_to_idx["Calanoid_1"]] = 1/3
                label_vector[self.class_to_idx["Cyclopoid_1"]] = 1/3
                label_vector[self.class_to_idx["Herpacticoida"]] = 1/3
                
            else:
                # Hard label (probability 1)
                label_vector[self.class_to_idx[category_name]] = 1.0
            self.labels.append(label_vector)
            

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
        label = torch.tensor(label,dtype=torch.float32)

        return image, label    
            
        
        
class TarImageDataset(Dataset):
    def __init__(self, folder_path,  transform=None,name_classes=None,
                 resolution: int = 28):
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
        self.tar_files = {}  # Cache for opened tar files
        
        if self.name_classes is None:
            name_classes = os.listdir(folder_path)

        # Load all image paths and their class names
        for class_name in name_classes:#os.listdir(folder_path):
            class_path = os.path.join(folder_path, class_name)
            #if os.path.isdir(class_path):  # Only consider directories
            #for image_name in os.listdir(class_path):
            with tarfile.open(class_path, 'r') as tar:
                for member in tar.getmembers():  
                    if member.name.lower().endswith('.tif'):  # Only load .tif files
                        self.image_files.append(os.path.join(class_path, member.name.rsplit("/",1)[-1]))
                        #self.image_files.append((class_path, member.name))
                        self.labels.append(class_name)

        # Create a mapping of class names to integer indices
        self.class_to_idx = {class_name.removesuffix(".tar"): idx for idx, class_name in enumerate(sorted(set(self.labels)))}
        self.labels = [self.class_to_idx[label.removesuffix(".tar")] for label in self.labels]  # Convert labels to integers

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        complete_path = self.image_files[idx]
        tar_path, image_filename = os.path.split(complete_path)  
        tar_folder = os.path.splitext(os.path.basename(tar_path))[0]  # Extract folder name from tar
        img_path = os.path.join(tar_folder, image_filename)
        
         # Open and cache tar file if not already opened
        if tar_path not in self.tar_files:
            self.tar_files[tar_path] = tarfile.open(tar_path, 'r')
        
        tar = self.tar_files[tar_path]
        file = tar.extractfile(img_path)    
        if file:    
            image = Image.open(io.BytesIO(file.read()))
                
        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)        
                

        # Get label
        label = self.labels[idx]

        # Convert label to a tensor
        label = torch.tensor(label, dtype=torch.long)
        
        return image, label    
    
    def close(self):
        """ Close all cached tar files to free resources. """
        for tar in self.tar_files.values():
            tar.close()
        self.tar_files.clear() 
        
class FilteredImageDataset(ImageDataset):
    def __init__(self, original_dataset, keep_indices):
        """
        Creates a new ImageDataset containing only a subset of the original images.
        
        Args:
            original_dataset (ImageDataset): The original dataset.
            keep_indices (list[int]): Indices of images to keep.
        """
        # Copy attributes from the original dataset
        self.folder_path = original_dataset.folder_path
        self.transform = original_dataset.transform
        self.name_classes = original_dataset.name_classes
        self.resolution = original_dataset.resolution
        self.class_to_idx = original_dataset.class_to_idx  # Keep class mapping

        # Subset the image paths and labels
        self.image_files = [original_dataset.image_files[i] for i in keep_indices]
        self.labels = [original_dataset.labels[i] for i in keep_indices]

    def __len__(self):
        return len(self.image_files)
    
            
# class FilteredImageDataset(Dataset):
#     def __init__(self, original_dataset, keep_indices):
#         """
#         Creates a filtered dataset from an existing ImageDataset.
        
#         Args:
#             original_dataset (ImageDataset): The original dataset.
#             keep_indices (list[int]): Indices of images to keep.
#         """
#         self.original_dataset = original_dataset
#         self.keep_indices = keep_indices  # Indices of images to keep

#     def __len__(self):
#         return len(self.keep_indices)

#     def __getitem__(self, idx):
#         # Map new index to original dataset
#         original_idx = self.keep_indices[idx]
#         return self.original_dataset[original_idx]

    

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


# def get_predictions (model,image,label):
    
#     with torch.no_grad():  
#     #model.eval()
#         out = model(image)
#         y_pred=out.argmax(dim=1).numpy()
            
#         true = label.numpy()
#         predicted = y_pred
#         print(image)
#     return true, predicted


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
