import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
import os

class Scene_Detection():
    def __init__(self, model, param, trained_model = None):

        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            self.model = model.to(self.device)
        else:
            self.model = model

        if trained_model is not None:
            file_path = os.path.join(param['model_save_dir'], trained_model)
            if os.path.isfile(file_path):
                checkpoint = torch.load(file_path)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print("=> loaded checkpoint '{}'".format(file_path))
            else:
                print("=> no checkpoint found at '{}'".format(file_path))


        self.model.eval()

        self.data_dir = param['data_dir']
        self.img_size = param['img_size']
        self.batch_size = param['batch_size']
        self.pic_save_dir = param['pic_save_dir']

        if not os.path.exists(self.pic_save_dir):
                os.makedirs(self.pic_save_dir)
    
    def data_transforms(self):
        SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
        SetScale = transforms.Lambda(lambda X: X/X.sum(0).expand_as(X))

        
        transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.CenterCrop(148),
                                        transforms.Resize(self.img_size),
                                        transforms.ToTensor(),
                                        SetRange])
        return transform

    def load_data(self):
        transform_list = self.data_transforms()
        dataset = datasets.ImageFolder(self.data_dir, transform=transform_list)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=4)
        self.data_loader = data_loader

    def get_latent_space(self):
        mu = None
        logvar = None
        label = None
        self.load_data()
        with torch.no_grad():
            for _, (x, y) in enumerate(self.data_loader):
                x = x.to(self.device)
                output = self.model.encode(x)
                latent_u, latent_logvar = output
                if mu == None and logvar == None:
                    mu = latent_u
                    logvar = latent_logvar
                    label = y
                else:
                    mu = torch.cat((mu, latent_u))
                    logvar = torch.cat((logvar, latent_logvar))
                    label = torch.cat((label, y))

        return mu.cpu().detach().numpy(), logvar.cpu().detach().numpy(), label


    def KL_latent(self, mu1, logvar1, mu2, logvar2):
        latent_dim = mu1.shape[0]
        sigma1 = np.exp(0.5*logvar1)
        sigma2 = np.exp(0.5*logvar2)
        KL = 0.5*((sigma1 / sigma2).sum() + ((mu2 - mu1) * (mu2 - mu1) / sigma2).sum() - latent_dim + np.log(sigma2.prod() / sigma1.prod()))
        return KL


    def compute_scene_KL_divergence(self):
        mu, logvar, label = self.get_latent_space()
        scene = np.unique(label)
        kl_full = []
        for i in scene:
            mu_temp, logvar_temp = mu[label == i], logvar[label == i]
            num_pic = mu_temp.shape[0]
            kl_list = []
            for j in range(1, num_pic):
                kl = self.KL_latent(mu_temp[j-1], logvar[j-1], mu_temp[j], logvar[j])
                kl_list.append(kl)
            #print(i, kl_list)
            kl_full.append(kl_list)
            plt.figure(figsize=(15, 3))
            plt.plot(list(range(num_pic-1)), kl_list)
            plt.savefig(self.pic_save_dir + '/scene' + str(i) + '.jpg')

        
        np.savetxt(self.pic_save_dir + '/scene_detect_kld.csv',
                   kl_full, delimiter = ',',
                   fmt ='% s',header = 'scene')