import time
from datetime import datetime

import torch
import torchvision.models as models
from torch.nn import functional as F
from modular import engine

class Model:

    def __init__(self, data_directory, num_classes, 
                 model_name: str = 'densenet121', device: torch.device = None):
        
        # Main class initializations
        self.model_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.data_directory = data_directory
        self.model_name = model_name
        self.device = device

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
            self.model.classifier = torch.nn.Linear(self.model.classifier.in_features, num_classes)
        elif self.model_name == 'resnet50':
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)

        # Other class initializations
        self.hyperparameters = None
        self.train_results = None


    def train(self, train_loader, val_loader, hyperparameters: dict):

        self.hyperparameters = hyperparameters

        # Loss function
        if hyperparameters['loss_fn'] == 'CrossEntropyLoss':
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
        early_stopping = engine.EarlyStopping(
            patience = early_stop_criteria['patience'],
            delta = early_stop_criteria['delta']
        )

        # Main training loop
        print(f'Starting training! Model: {self.model_name} (ID: {self.model_id})')

        start = time.time()
        self.train_results = engine.train_test_loop(
            model = self.model,
            train_dataloader = train_loader,
            test_dataloader = val_loader,
            optimizer = optimizer,
            loss_fn = loss_fn,
            epochs = hyperparameters['epochs'],
            Scheduler = scheduler,
            early_stopping = early_stopping,
            device = self.device,
            print_b = True
        )

        elapsed = time.time() - start
        print(f'Training Finished! Time Elapsed: {elapsed} sec.')


    def predict(self, test_loader):

        self.model.eval()
        labels, probs, preds = [], [], []

        with torch.no_grad():
            for image, label in test_loader:
                image = image.to(self.device)
                label = image.to(self.device)
                
                output = self.model(image)
                prob = F.softmax(output, dim = 1)
                pred = output.argmax(dim = 1)

            labels.append(label)
            probs.append(prob)
            preds.append(pred)

        return torch.cat(labels), torch.cat(probs), torch.cat(preds)
    
