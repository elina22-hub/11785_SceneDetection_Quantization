import os
from typing import List, Callable, Union, Any, TypeVar, Tuple
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
Tensor = TypeVar('torch.tensor')
from tqdm.notebook import tqdm
from pytorch_lightning.loggers.test_tube import TestTubeLogger
import numpy as np
import matplotlib.pyplot as plt
import datetime

class Trainer():
    def __init__(self, model, params:dict, trained_model = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            self.model = model.to(self.device)
        else:
            self.model = model

        self.lr = params['learning_rate']
        self.batch_size = params['batch_size']
        self.data_dir = params['data_dir']
        self.model_save_dir = params['model_save_dir']
        self.log_save_dir = params['log_save_dir']
        self.log_save_name = params['log_save_name']
        self.img_size = params['img_size']
        self.scheduler_gamma = params['scheduler_gamma']
        self.weight_decay = params['weight_decay']
        self.max_epochs = params['max_epochs']
        self.global_epochs = 0

        torch.manual_seed(params['manual_seed'])
        np.random.seed(params['manual_seed'])
        
        self.train_data_loaded = False
        self.val_data_loaded = False
        self.optimizer = optim.Adam(self.model.parameters(),lr = self.lr,weight_decay = self.weight_decay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer,self.scheduler_gamma)
        self.best_val_total_loss = float('inf')

        self.logger = TestTubeLogger(save_dir=self.log_save_dir,
                                     name=self.log_save_name,
                                     debug=False,
                                     create_git_tag=False)

        if trained_model is not None:
            self.load_checkpoint(trained_model)
        
        if not os.path.exists(self.model_save_dir):
                os.makedirs(self.model_save_dir)
        if not os.path.exists(self.log_save_dir):
                os.makedirs(self.log_save_dir)

    def data_transforms(self):
        transform = transforms.Compose([transforms.Resize((self.img_size, self.img_size)),
                                        transforms.ToTensor()])
        return transform

    def load_data(self, phase):
        transform_list = self.data_transforms()
        dataset = datasets.ImageFolder(self.data_dir, transform=transform_list)
        if phase == 'train':
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False, num_workers=4)
        else:
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=4)
        return data_loader  
    
    def train_dataloader(self):
        self.train_data_loaded = True
        return self.load_data('train')

    def val_dataloader(self):
        self.val_data_loaded = True
        return self.load_data('val')

    def train_model(self):
        self.model.train()
        train_loss = {'train_total_loss':0.0, 'train_recon_loss':0.0,'train_kld':0.0}
        len_data_loader = len(self.train_dataloader) 
        for _, (x, y) in tqdm(enumerate(self.train_dataloader),total=len_data_loader,leave = False):
            self.optimizer.zero_grad()     
            x= x.to(self.device)
            results = self.model.forward(x, labels=y)
            loss = self.model.loss_function(*results)

            loss['total_loss'].backward()
            self.optimizer.step()

            train_loss['train_total_loss'] += loss['total_loss'].cpu().detach().numpy()
            train_loss['train_recon_loss'] += loss['recon_loss'].cpu().detach().numpy()
            train_loss['train_kld'] += loss['kld'].cpu().detach().numpy()
            
        train_loss['train_total_loss'] /= len_data_loader
        train_loss['train_recon_loss'] /= len_data_loader
        train_loss['train_kld'] /= len_data_loader
        
        return train_loss

    def val_model(self):
        self.model.eval()
        val_loss = {'val_total_loss':0.0, 'val_recon_loss':0.0,'val_kld':0.0}
        len_data_loader = len(self.val_dataloader) 
        with torch.no_grad():
            for _, (x, y) in tqdm(enumerate(self.val_dataloader),total=len_data_loader, leave = False):
                x = x.to(self.device)
                results = self.model.forward(x)
                loss = self.model.loss_function(*results)

                val_loss['val_total_loss'] += loss['total_loss'].cpu().detach().numpy()
                val_loss['val_recon_loss'] += loss['recon_loss'].cpu().detach().numpy()
                val_loss['val_kld'] += loss['kld'].cpu().detach().numpy()
            
        val_loss['val_total_loss'] /= len_data_loader
        val_loss['val_recon_loss'] /= len_data_loader
        val_loss['val_kld'] /= len_data_loader

        return val_loss
    
    def init_train_config(self):
        if not self.train_data_loaded:
            self.train_dataloader = self.load_data('train')
            self.train_data_loaded = True
        if not self.val_data_loaded:
            self.val_dataloader = self.load_data('val')
            self.train_data_loaded = True
    
    def save_checkpoint(self, filename = 'Model'):
        filename += '_'+f"{datetime.datetime.now():%Y%m%d}"
        file_path = os.path.join(self.model_save_dir, filename)
        torch.save({'global_epochs': self.global_epochs,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict' : self.scheduler.state_dict(),
                    'best_val_total_loss':self.best_val_total_loss
                    }, file_path)
        print("=> save new model '{}'".format(file_path))

    def load_checkpoint(self,filename):
        file_path = os.path.join(self.model_save_dir, filename)
        if os.path.isfile(file_path):
            checkpoint = torch.load(file_path)
            self.global_epochs = checkpoint['global_epochs']
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("=> loaded checkpoint '{} (epoch {})'".format(file_path, self.global_epochs))
        else:
            print("=> no checkpoint found at '{}'".format(file_path))

    
    def fit(self):
        self.init_train_config()
        while True:
            self.global_epochs += 1
            train_loss = self.train_model()
            val_loss = self.val_model()

            print('------Epoch {}------'.format(self.global_epochs))
            print('Train: total_loss:{:6f}, recon_loss:{:6f}, kld:{:6f}'.format(train_loss['train_total_loss'],train_loss['train_recon_loss'],train_loss['train_kld']))
            print('Val  : total_loss:{:6f}, recon_loss:{:6f}, kld:{:6f}'.format(val_loss['val_total_loss'],val_loss['val_recon_loss'],val_loss['val_kld']))
            if (val_loss['val_total_loss'] < self.best_val_total_loss):
                self.best_val_total_loss = val_loss['val_total_loss']
                self.save_checkpoint()
            if self.global_epochs >= self.max_epochs:
                break
            
            train_loss.update(val_loss)
            self.logger.log_metrics(train_loss)
            self.logger.save()
                
            self.scheduler.step()
    