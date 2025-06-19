import os
import copy
import random

from PIL import Image
from collections import Counter

import torch
from torch.utils.data import Dataset, Subset, DataLoader, SequentialSampler, WeightedRandomSampler, random_split
from torchvision import transforms

from helper_functions import set_seed


class ImageDataset(Dataset):

    """
    A custom PyTorch Dataset for loading and preprocessing image data from a directory
    where each subfolder represents a class.

    This class handles:
    - Class-wise and random sampling with optional size limits
    - Optional image transforms for data augmentation
    - Preprocessing and setup for imbalanced class handling

    Args:
        data_directory (str): Path to the root dataset directory. Each subdirectory should represent a class.
        class_names (list, optional): List of class names to include. If None, all subdirectories are included.
        class_sizes (list, optional): Number of samples to include per class. If None, uses `max_class_size` for all.
        max_class_size (int, optional): Default maximum number of samples to draw per class. Defaults to 10,000.
        image_resolution (int, optional): Final size (height and width) to resize images to. Defaults to 28.
        image_transforms (callable, optional): Image transformations (e.g., data augmentations) to apply. Defaults to None.
        seed (int, optional): Random seed for reproducibility. Defaults to 666.

    Attributes:
        data_directory (str): Path to the dataset root directory.
        seed (int): Random seed used for sampling.
        class_names (list): Sorted list of class names included in the dataset.
        class_sizes (torch.Tensor): Tensor of the actual sampled size per class.
        class_indices (list): Numeric index for each class (aligned with `class_names`).
        image_paths (list): List of file paths to all sampled images.
        labels (list): List of numeric class IDs corresponding to each image.
        image_resolution (int): Size to which each image is resized.
        image_transforms (callable or None): Image transformations applied during training or inference.
    """

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

        """
        Returns the number of samples in the Dataset.
        """

        return len(self.image_paths)
    

    def __getitem__(self, idx):

        """
        Returns the image and label of specified sample.

        Args:
            idx (int): Index of specified sample.
        """
        
        image = Image.open(self.image_paths[idx]).convert('L')
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.image_transforms:
            image = self.image_transforms(image)

        image = image.repeat(3, 1, 1)
        
        return image, label
    
    
    def print_dataset_details(self, indices: list = None, subset_name: str = None):

        """
        Prints the class distribution of the Dataset.

        Args:
            indices (list, optional): Specific indices of subset of Dataset to consider.
            subset_name (str, optional): Specific name of subset of Dataset to print.
        """

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


    def print_image_transforms(self):

        """
        Prints the ordered image transformations applied to the Dataset.
        """

        print('\nCurrent Image Transform Pipeline:')
        for tf in self.image_transforms.transforms:
            print(' ', tf)


    def filter_to_class(self, class_id: int):

        """
        Returns the Dataset filtered to a specified class as a Subset object.

        Args:
            class_id (int): ID of class to filter Dataset to.
        """

        filtered_idx = torch.where(torch.tensor(self.labels) == class_id)[0].tolist()

        return Subset(self, filtered_idx)
    

    def subsample_classes(self, subsample_sizes: dict):

        """
        Samples each class from the Dataset and returns a Subset object.

        Args:
            subsample_sizes (dict): Key-value pairs of class label and specified sample size.
                - Maximum number of items in dict is number of classes in Dataset.
        """

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
    
    
    def compute_sample_weights(self, indices: list = None, weights: str = 'inverse_weighted', normalize_weights: bool = True):

        """
        Computes weights per class in the Dataset and assigns each sample the corresponding class weight.

        Returns `sample_weights` of length equal to number of samples and 
            `class_weights` of length equal to number of classes.

        Args:
            indices (list, optional): Specific indices of subset of Dataset to consider.
            weights (str): Specifies which weights computation to use.
            normalize_weights (bool): Specifies if weights should be normalized.
        """

        labels = torch.tensor(self.labels, dtype = torch.long)
        if indices is not None:
            sub_labels = labels[indices]
        else:
            sub_labels = labels

        class_counts = torch.bincount(sub_labels, minlength = len(self.class_names)).float()

        if weights == 'inverse_weighted':
            class_weights = len(self) / (class_counts * len(self.class_names))
        elif weights == 'inverse':
            class_weights = 1.0 / class_counts
        elif weights == 'normalized':
            class_weights = class_counts / class_counts.sum()

        if normalize_weights:
            class_weights = class_weights / class_weights.sum()

        sample_weights = class_weights[sub_labels]

        return sample_weights, class_weights
    

    def split_train_test_val(self, train_prop: float = 0.7, val_prop: float = 0.1, test_prop: float = 0.2, verbose: bool = True):

        """
        Returns indices corresponding to the train, validation and test subsets of the Dataset.

        Args:
            trian_prop (float): Proportion of samples to allocate to the train subset.
            val_prop (float): Proportion of samples to allocate to the validation subset.
            test_prop (float): Proportion of samples to allocate to the test subset.
            verbose (bool): Specifies whether to print distributions of subsets.
        """

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
                                replace: bool = False, verbose: bool = False):
        
        """
        Appends image transformations to existing transformation pipeline or replaces.
        If multiple `ToTensor()` transformations are included in the resulting pipeline, only the last instance is kept.
        If there are no `ToTensor()` transformations in the resulting pipeline, it is appended.

        Args:
            image_transforms(transfors.Compose, optional): Iterable of image transformations to append.
            replace (bool): Specifies whether to replace with or append the above image_transforms.
            verbose (bool): Specifies whether to print the resulting image transformation pipeline.
        """
        
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
            self.print_image_transforms()
    

    def create_dataloaders(self, batch_size: int, train_indices, val_indices, test_indices,
                           image_transforms: transforms.Compose = None, transform_val: bool = False, 
                           train_sample_weights: torch.tensor = None):
        
        """
        Creates the train, validatinon and test DataLoaders required for training a PyTorch model.
        If `train_sample_weights` is specified, they are supplied to WeightedRandomSampler for the train subset.

        Args:
            batch_size (int): Sizes of batches to process samples in DataLoader.
            train_indices (list): Indices corresponding to the train subset of the Dataset.
            val_indices (list): Indices corresponding to the validation subset of the Dataset.
            test_indices (list): Indices corresponding to the test subset of the Dataset.
            image_transforms (transforms.Compose, optional): Additional image transformations for the train subset.
            transform_val (bool): Specifies whether to apply train image transformations to the validation subset.
            train_sample_weights (torch.tensor, optional): Contains weights for each sample in the train subset.
        """

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

        if train_sample_weights is None:
            train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, 
                generator = torch.Generator().manual_seed(self.seed)
            )
        else:
            train_loader = DataLoader(
                train_dataset, batch_size = batch_size, 
                sampler = WeightedRandomSampler(train_sample_weights, num_samples = len(train_sample_weights), replacement = True)
            )

        val_loader = DataLoader(
            val_dataset, batch_size = batch_size, sampler = SequentialSampler(val_dataset)
        )
        test_loader = DataLoader(
            test_dataset, batch_size = batch_size, sampler = SequentialSampler(test_dataset)
        )

        return train_loader, val_loader, test_loader

        