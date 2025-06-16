import time
from datetime import datetime
from itertools import product

import torch
import torchvision.models as models
from torch.nn import functional as F
from modular import engine

from helper_functions import set_seed, extract_metrics

class Model:

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
        else:
            raise ValueError('Unsupported scheduler. Select one of StepLR.')
        
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


    def gridsearch(self, parameter_grid: dict, train_loader, val_loader, scoring_fn, scoring: str = 'accuracy'):

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

