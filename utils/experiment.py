from typing import List, Callable, Union, Any, TypeVar, Tuple
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
import pytorch_lightning as pl
Tensor = TypeVar('torch.tensor')


class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model,
                 params: dict) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.val_best_loss = float('inf')
        self.best_model_epoch = None

        self.num_epoch = 1
        self.full_log = {}

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        train_loss = self.model.loss_function(*results,
                                              M_N=self.params['batch_size'] / self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx=batch_idx)
        # self.logger.experiment.log({key: val.item() for key, val in train_loss.items()})
        return train_loss

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        # print(avg_loss)
        avg_loss = avg_loss.cpu().detach().numpy() / self.params['batch_size']
        # print(type(avg_loss))
        self.log_summary = {'avg_train_loss': avg_loss}
        # self.logger.experiment.log(log)

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        val_loss = self.model.loss_function(*results,
                                            M_N=self.params['batch_size'] / self.num_val_imgs,
                                            optimizer_idx=optimizer_idx,
                                            batch_idx=batch_idx)

        return val_loss

    def validation_epoch_end(self, outputs):

        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        # print(avg_loss)

        avg_loss = avg_loss.cpu().detach().numpy() / 1.0
        # print(type(avg_loss))

        self.log_summary['avg_val_loss'] = avg_loss
        print('epoch', self.num_epoch, self.log_summary)

        self.full_log[self.num_epoch] = self.log_summary
        self.num_epoch += 1

        self.logger.experiment.log(self.log_summary)

        if avg_loss < self.val_best_loss:
            torch.save(self.model.state_dict(), './best_model/model')
            self.val_best_loss = avg_loss
            self.best_model_epoch = self.num_epoch
            print('Save new model')

        # del self.log_summary

    def sample_images(self):
        # Get sample reconstruction image
        test_input, test_label = next(iter(self.sample_dataloader))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)
        recons = self.model.generate(test_input, labels=test_label)
        vutils.save_image(recons.data,
                          f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
                          f"recons_{self.logger.name}_{self.current_epoch}.png",
                          normalize=True,
                          nrow=12)

        # vutils.save_image(test_input.data,
        #                   f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
        #                   f"real_img_{self.logger.name}_{self.current_epoch}.png",
        #                   normalize=True,
        #                   nrow=12)

        try:
            samples = self.model.sample(144,
                                        self.curr_device,
                                        labels=test_label)
            vutils.save_image(samples.cpu().data,
                              f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
                              f"{self.logger.name}_{self.current_epoch}.png",
                              normalize=True,
                              nrow=12)
        except:
            pass

        del test_input, recons  # , samples

    def configure_optimizers(self):
        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)

        scheduler = optim.lr_scheduler.ExponentialLR(optims[0], gamma=self.params['scheduler_gamma'])
        scheds.append(scheduler)

        return optims, scheds

    def train_dataloader(self):
        transform = self.data_transforms()
        dataset = datasets.ImageFolder(self.params['data_dir'], transform=transform)
        self.num_train_imgs = len(dataset)
        return DataLoader(dataset,
                          batch_size=self.params['batch_size'],
                          num_workers=4,
                          shuffle=True,
                          drop_last=True,
                          pin_memory=True)

    def val_dataloader(self):
        transform = self.data_transforms()
        dataset = datasets.ImageFolder(self.params['data_dir'], transform=transform)

        self.sample_dataloader = DataLoader(dataset,
                                            batch_size=1,
                                            num_workers=4,
                                            shuffle=False,
                                            drop_last=True,
                                            pin_memory=True)

        self.num_val_imgs = len(self.sample_dataloader)
        return self.sample_dataloader

    def data_transforms(self):
        SetRange = transforms.Lambda(lambda X: 2 * X - 1.)   # why?
        SetScale = transforms.Lambda(lambda X: X / X.sum(0).expand_as(X))

        transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.CenterCrop(148),
                                        transforms.Resize(self.params['img_size']),
                                        transforms.ToTensor(),
                                        SetRange])
        return transform