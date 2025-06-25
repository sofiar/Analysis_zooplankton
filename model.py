import time
from datetime import datetime
from itertools import product

import torch
import torchvision.models as models
from torch.nn import functional as F
from modular import engine

from helper_functions import set_seed, extract_metrics

class Model:

    """
    A wrapper for training and evaluating image classification models using 
    DenseNet121 or ResNet50 models.

    This class handles:
    - Model initialization with pretrained weights
    - Replacement of the classifier head for custom number of classes
    - Initializes the model instance

    Args:
        data_directory (str): Path to the directory containing model weights.
        num_classes (int): Number of output classes for classification.
        model_name (str, optional): Model to use ('densenet121' or 'resnet50'). Defaults to 'densenet121'.
        device (torch.device, optional): Computation device (e.g., torch.device('cuda')). Defaults to None.
        seed (int, optional): Random seed for reproducibility. Defaults to 666.

    Attributes:
        model_id (str): Unique identifier for the model instance (based on datetime).
        data_directory (str): Path to model weights and data.
        num_classes (int): Number of classification labels.
        model_name (str): Architecture used by the model.
        device (torch.device): Device on which the model runs.
        seed (int): Random seed used for reproducibility.
        model (torch.nn.Module): The PyTorch model instance with the custom classification head.
        weights_path (str): File path to the pretrained weights.
        hyperparameters (dict or None): Dictionary of training hyperparameters (set later).
        train_results (dict or None): Stores training and evaluation metrics (set later).
    """

    def __init__(self, data_directory, num_classes, model_name: str = 'densenet121', 
                 device: torch.device = None, seed: int = 666):
        
        # Main class initializations
        self.model_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.data_directory = data_directory
        self.num_classes = num_classes
        self.model_name = model_name
        self.device = device
        self.seed = seed
        
        set_seed(self.seed)

        # Load model and weights
        if self.model_name == 'densenet121':
            self.model = models.densenet121(weights = None)
            self.weights_path = self.data_directory + '/densenet121-a639ec97.pth'
        elif self.model_name == 'resnet50':
            self.model = models.resnet50(weights = None)
            self.weights_path = self.data_directory + '/resnet50-0676ba61.pth'
        else:
            raise ValueError('Unsupported model. Select one of densenet121 or resnet50.')
        
        state_dict = torch.load(self.weights_path, map_location = 'cpu')
        self.model.load_state_dict(state_dict, strict = False)
        self.model.to(self.device)

        if self.model_name == 'densenet121':
            self.model.classifier = torch.nn.Linear(self.model.classifier.in_features, self.num_classes)
        elif self.model_name == 'resnet50':
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, self.num_classes)

        # Other class initializations
        self.hyperparameters = None
        self.train_results = None


    def train(self, hyperparameters: dict, train_loader, val_loader, verbose: bool = True):

        """
        Trains the model instance using the specified data loaders and training parameters.
        Not all training parameters are supported. Sample hyperparameters dictionary:
            HYPERPARAMETERS = {
                'loss_fn': {'type': 'CrossEntropyLoss', 'weights': train_class_weights}, 
                'optimizer': 'Adam', 
                'lr': 5e-4, 
                'epochs': 40, 
                'scheduler': {'type': 'StepLR', 'step_size': 10, 'gamma': 0.1}, 
                'early_stopping': {'patience': 10, 'delta': 0.005}
            }

        Args:
            hyperparameters (dict): Dictionary containing training parameters such as loss function, 
                                    optimizer type, learning rate, scheduler settings, early stopping, and epochs.
            train_loader (DataLoader): PyTorch DataLoader containing the training dataset.
            val_loader (DataLoader): PyTorch DataLoader containing the validation dataset.
            verbose (bool, optional): Whether to print training progress. Defaults to True.
        """

        set_seed(self.seed)

        self.hyperparameters = hyperparameters

        # Loss function
        loss_fn_spec = hyperparameters['loss_fn']
        if loss_fn_spec['type'] == 'CrossEntropyLoss':
            if loss_fn_spec['weights'] is not None:
                loss_fn = torch.nn.CrossEntropyLoss(weight = loss_fn_spec['weights'].to(self.device))
            else:
                loss_fn = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError('Unsupported loss function. Select one of CrossEntropyLoss.')
        
        # Optimizer
        if hyperparameters['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(
                params = self.model.parameters(), lr = hyperparameters['lr']
            )
        elif hyperparameters['optimizer'] == 'SGD':
            optimizer = torch.optim.SGD(
                params = self.model.parameters(), lr = hyperparameters['lr']
            )
        else:
            raise ValueError('Unsupported optimizer. Select one of Adam or SGD.')
        
        # Scheduler
        scheduler_spec = hyperparameters['scheduler']
        if scheduler_spec['type'] == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size = scheduler_spec['step_size'], gamma = scheduler_spec['gamma']
            )
        elif scheduler_spec['type'] == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max = scheduler_spec['T_max']
            )
        elif scheduler_spec['type'] == 'OneCycleLR':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr = scheduler_spec['max_lr'], epochs = hyperparameters['epochs'], steps_per_epoch = len(train_loader)
            )
        else:
            raise ValueError('Unsupported scheduler. Select one of StepLR, CosineAnnealingLR or OneCycleLR.')
        
        # Early stopping
        early_stop_criteria = hyperparameters['early_stopping']
        if early_stop_criteria is not None:
            early_stopping = engine.EarlyStopping(
                patience = early_stop_criteria['patience'],
                delta = early_stop_criteria['delta']
            )
        else:
            early_stopping = None

        # Main training loop
        if verbose:
            print(f'\nStarting training! Model: {self.model_name} (ID: {self.model_id})\n')

        start = time.time()
        train_results = engine.train_test_loop(
            model = self.model,
            train_dataloader = train_loader,
            test_dataloader = val_loader,
            optimizer = optimizer,
            loss_fn = loss_fn,
            epochs = hyperparameters['epochs'],
            Scheduler = scheduler,
            early_stopping = early_stopping,
            device = self.device,
            print_b = verbose
        )
        self.train_results = extract_metrics(train_results)

        elapsed = time.time() - start
        if verbose:
            print(f'\nTraining Finished! Time Elapsed: {elapsed:.2f} sec.')


    def predict(self, test_loader):

        """
        Generates predictions for the specified samples using the trained model instance.

        Args:
            test_loader (DataLoader): PyTorch DataLoader containing the test dataset.
        """

        self.model.eval()
        labels, probs, preds = [], [], []

        with torch.no_grad():
            for image, label in test_loader:
                image = image.to(self.device)
                label = label.to(self.device)
                
                output = self.model(image)
                prob = F.softmax(output, dim = 1)
                pred = output.argmax(dim = 1)

                labels.append(label)
                probs.append(prob)
                preds.append(pred)

        return torch.cat(labels), torch.cat(probs), torch.cat(preds)


    def gridsearch(self, parameter_grid: dict, train_loader, val_loader, scoring_fn):

        """
        Performs grid search over a set of hyperparameters to identify the best configuration.
        Sampler parameter grid to search:
            HYPERPARAMETER_SEARCH_GRID = {
                'loss_fn': [
                    {'type': 'CrossEntropyLoss', 'weights': None},
                    {'type': 'CrossEntropyLoss', 'weights': train_class_weights},
                ],  
                'optimizer': ['Adam'],
                'lr': [1e-3, 5e-4, 1e-4],
                'epochs': [40],
                'scheduler': [
                    {'type': 'StepLR', 'step_size': 10, 'gamma': 0.1},
                ],
                'early_stopping': [
                    {'patience': 10, 'delta': 0.005},
                ],
            }

        Args:
            parameter_grid (dict): Dictionary where keys are hyperparameter names and values are lists of values to try.
            train_loader (DataLoader): PyTorch DataLoader for the training dataset.
            val_loader (DataLoader): PyTorch DataLoader for the validation dataset.
            scoring_fn (callable): A function to evaluate model predictions (e.g., accuracy_score or f1_score).
        """

        parameters = list(parameter_grid.keys())

        best_score = float('-inf')
        best_params = None

        print(f'\nStarting hyperparameter tuning!')
    
        start = time.time()
        for iter, param_values in enumerate(product(*parameter_grid.values())):

            current_parameters = dict(zip(parameters, param_values))
            print(f'\nStarting Iteration {iter+1}!')
            print(f'Parameters: {current_parameters}')

            start_iter = time.time()

            # Initiate a new Model object and train based on current parameters
            current_model = Model(
                data_directory = self.data_directory,
                num_classes = self.num_classes,
                model_name = self.model_name,
                device = self.device,
                seed = self.seed
            )

            current_model.train(
                hyperparameters = current_parameters,
                train_loader = train_loader,
                val_loader = val_loader,
                verbose = False
            )

            # Get predictions and score on validation set
            labels, _, preds = current_model.predict(val_loader)
            score = scoring_fn(labels.cpu(), preds.cpu())

            elapsed_iter = time.time() - start_iter
            print(f'Completed Iteration {iter+1}! Time Elapsed {elapsed_iter:.2f} sec. | Score: {score:.4f}')

            # Update parameters if necessary
            if score > best_score:
                best_score = score
                best_params = current_parameters
            
            del current_model
        
        elapsed = time.time() - start
        print(f'\nHyperparameter Tuning Finished! Total Time Elapsed: {elapsed:.2f} sec.')

        return best_params, best_score

