import torch
import matplotlib.pyplot as plt
import numpy as np
# Calculate KL divergence

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

def get_latent_space(model, dataloader):
    model.to(device)
    model.eval()

    mu = None
    logvar = None
    label = None
    with torch.no_grad():
        for _, (x, y) in enumerate(dataloader):
            x = x.to(device)
            output = model.encode(x)
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


def KL_latent(mu1, logvar1, mu2, logvar2):
    latent_dim = mu1.shape[0]
    sigma1 = np.exp(0.5*logvar1)
    sigma2 = np.exp(0.5*logvar2)
    KL = 0.5*((sigma1 / sigma2).sum() + ((mu2 - mu1) * (mu2 - mu1) / sigma2).sum() - latent_dim + np.log(sigma2.prod() / sigma1.prod()))
    return KL


def scene_KL_divergence(model, dataloader):
    mu, logvar, label = get_latent_space(model, dataloader)
    scene = np.unique(label)
    for i in scene:
        mu_temp, logvar_temp = mu[label == i], logvar[label == i]
        num_pic = mu_temp.shape[0]
        kl_list = []
        for j in range(1, num_pic):
            kl = KL_latent(mu_temp[j-1], logvar[j-1], mu_temp[j], logvar[j])
            kl_list.append(kl)
        print(i, kl_list)
        plt.figure(figsize=(15, 3))
        plt.plot(list(range(num_pic-1)), kl_list)
        plt.savefig('./scene_divergence_graph/scene' + str(i) + '.jpg')