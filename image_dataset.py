import os
import copy
import random

from PIL import Image
from collections import Counter

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from torch.utils.data import random_split, Subset, DataLoader, SequentialSampler

from helper_functions import set_seed


class ImageDataset(Dataset):

    def __init__(self, data_directory, 
                 class_names: list = None, class_sizes: list = None, max_class_size: int = 10000, 
                 image_resolution: int = 28, image_transforms = None, seed: int = 666):
        
        self.data_directory = data_directory
        self.seed = seed

        set_seed(seed)

        # Specify subset of classes to consider; all classes considered if None
        if class_names is None:
            class_names = sorted(os.listdir(self.data_directory))

        # Specify initial number of samples to consider per class; max if None
        if class_sizes is None:
            class_sizes = [max_class_size] * len(class_names)
        
        self.class_names, self.class_sizes = map(list, zip(*sorted(zip(class_names, class_sizes))))
        self.class_indices = list(range(len(self.class_names)))

        # Iterate through each class and sample .tif images only; append paths and labels
        self.image_paths = []
        self.labels = []

        for class_id, class_name in zip(self.class_indices, self.class_names):
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
        self.image_transforms = image_transforms
        
    
    def __len__(self):

        return len(self.image_paths)
    

    def __getitem__(self, idx):
        
        image = Image.open(self.image_paths[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.image_transforms:
            image = self.image_transforms(image)

        return image, label
    
    
    def print_dataset_details(self, indices: list = None, subset_name: str = None):

        if indices is None:
            filtered_labels = self.labels
        else:
            filtered_labels = [self.labels[i] for i in indices]

        filtered_counts = dict(Counter(filtered_labels))

        if subset_name is None:
            print(f'\nTotal Dataset Size: {len(filtered_labels)}')
        else:
            print(f'\n{subset_name} Dataset Size: {len(filtered_labels)}')

        for class_id, class_name in zip(self.class_indices, self.class_names):
            class_prop = filtered_counts[class_id] / len(filtered_labels)

            print(f'Class Name: {class_name} | Class Label: {class_id} | Count: {filtered_counts[class_id]} ' +
                  f'| Prop: {class_prop:.2f}'
            )


    def filter_to_class(self, class_id: int):

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
    

    def split_train_test_val(self, train_prop: float = 0.7, val_prop: float = 0.1, test_prop: float = 0.2, verbose: bool = True):

        train_split, val_split, test_split = random_split(
            range(len(self)),
            lengths = [train_prop, val_prop, test_prop],
            generator = torch.Generator().manual_seed(self.seed)
        )

        if verbose:
            self.print_dataset_details(train_split.indices, 'Train')
            self.print_dataset_details(val_split.indices, 'Validation')
            self.print_dataset_details(test_split.indices, 'Test')

        return train_split.indices, val_split.indices, test_split.indices
    

    def append_image_transforms(self, image_transforms: transforms.Compose = None, 
                                replace: bool = False, verbose: bool = True):
        
        if not isinstance(image_transforms, transforms.Compose):
            raise TypeError('Unsupported type: image_transforms must be a torchvision.transforms.Compose object.')
        
        if self.image_transforms is None or replace:
            image_transforms_list = image_transforms.transforms
        else:
            image_transforms_list = self.image_transforms.transforms + image_transforms.transforms

        image_transforms_cleaned = []
        to_tensor_indices = [i for i, tf in enumerate(image_transforms_list) if isinstance(tf, transforms.ToTensor)]

        if to_tensor_indices:
            last_idx = to_tensor_indices[-1]
            image_transforms_cleaned = [tf for i, tf in enumerate(image_transforms_list) if not isinstance(tf, transforms.ToTensor) or i == last_idx]
        else:
            image_transforms_cleaned = image_transforms_list + [transforms.ToTensor()]

        self.image_transforms = transforms.Compose(image_transforms_cleaned)

        if verbose:
            print('\nCurrent Image Transform Pipeline:')
            for tf in self.image_transforms.transforms:
                print(' ', tf)
    

    def create_dataloaders(self, batch_size: int, train_indices, val_indices, test_indices, 
                           image_transforms: transforms.Compose = None, transform_val: bool = False):

        if image_transforms is not None:
            dataset_aug = copy.deepcopy(self)
            dataset_aug.append_image_transforms(
                image_transforms = image_transforms, verbose = False
            )
            train_dataset = Subset(dataset_aug, train_indices)
            if transform_val:
                val_dataset = Subset(dataset_aug, val_indices)
            else:
                val_dataset = Subset(self, val_indices)
        else:
            train_dataset = Subset(self, train_indices)
            val_dataset = Subset(self, val_indices)
        
        test_dataset = Subset(self, test_indices)

        train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, 
            generator = torch.Generator().manual_seed(self.seed)
        )
        val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = True, 
            generator = torch.Generator().manual_seed(self.seed)
        )
        test_loader = DataLoader(
            test_dataset, batch_size = batch_size, sampler = SequentialSampler(test_dataset)
        )

        return train_loader, val_loader, test_loader

        