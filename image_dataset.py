import os
import random

import numpy as np
from PIL import Image
from collections import Counter

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, Subset
from torchvision import transforms


class ImageDataset(Dataset):

    def __init__(self, data_directory, 
                 class_names: list = None, class_sizes: list = None, max_class_size: int = 10000, 
                 image_resolution: int = 28, image_transforms = None, seed: int = 666):
        
        self.data_directory = data_directory
        self.seed = seed

        # Specify subset of classes to consider; all classes considered if None
        if class_names is None:
            class_names = sorted(os.listdir(self.data_directory))

        # Specify initial number of samples to consider per class; max if None
        if class_sizes is None:
            class_sizes = [max_class_size] * len(class_names)
        
        self.class_names, self.class_sizes = map(list, zip(*sorted(zip(class_names, class_sizes))))
        self.class_idx = list(range(len(self.class_names)))

        # Iterate through each class and sample .tif images only; append paths and labels
        self.image_paths = []
        self.labels = []

        for class_id, class_name in zip(self.class_idx, self.class_names):
            class_directory = os.path.join(data_directory, class_name)

            if os.path.isdir(class_directory):
                class_paths = os.listdir(class_directory)
                new_class_size = min(self.class_sizes[class_id], len(class_paths))

                random.seed(self.seed)
                sampled_paths = random.sample(class_paths, new_class_size)

                for image_path in sampled_paths:
                    if image_path.lower().endswith('.tif'):
                        self.image_paths.append(os.path.join(class_directory, image_path))
                    else:
                        new_class_size -= 1

            self.class_sizes[class_id] = new_class_size
            self.labels.extend([class_id] * new_class_size)
        
        # Other class initializations
        self.image_resolution = image_resolution
        self.imgage_transforms = image_transforms
        
    
    def __len__(self):

        return len(self.image_paths)
    

    def __getitem__(self, idx):
        
        image = Image.open(self.image_paths[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.imgage_transforms:
            image = self.imgage_transforms(image)

        return image, label
    
    
    def print_dataset_details(self, indices: list = None):

        if indices is None:
            filtered_labels = self.labels
        else:
            filtered_labels = [self.labels[i] for i in indices]

        filtered_counts = dict(Counter(filtered_labels))

        print(f'Total Dataset Size: {len(filtered_labels)}\n')

        for class_id, class_name in zip(self.class_idx, self.class_names):
            print(f'Class Name: {class_name} | Class Label: {class_id} | Count: {filtered_counts[class_id]}')


    def filter_to_class(self, class_id):

        filtered_idx = torch.where(torch.tensor(self.labels) == class_id)[0].tolist()

        return Subset(self, filtered_idx)
    

    def subsample_classes(self, subsample_sizes: dict):

        random.seed(self.seed)

        all_sampled_idx = []

        for class_id, class_sample_size in subsample_sizes.items():
            filtered_idx = torch.where(torch.tensor(self.labels) == class_id)[0].tolist()

            if len(filtered_idx) > class_sample_size:
                sampled_idx = random.sample(filtered_idx, class_sample_size)
            else:
                sampled_idx = filtered_idx

            all_sampled_idx.extend(sampled_idx)
        
        return Subset(self, all_sampled_idx)
    
