# Deep Learning pour la gestion d'actif
# Vivienne Investissmeent 
#
# Training algorithm
#
# Thomas Beroud
# Aug, 2022

import torch
import numpy as np
from torch.autograd import grad as torch_grad
import os
from plot_result import Analysis_sf, Analysis_m
import matplotlib.pyplot as plt

from time import time


class Trainer():
    def __init__(self, generator, discriminator, gen_optimizer, disc_optimizer, batch_size, latent_dim, ts_dim, data_train, device, scorepath, gp_weight=10, critic_iter=5): 
        self.G = generator
        self.D = discriminator
        self.G_opt = gen_optimizer
        self.D_opt = disc_optimizer
        self.batch_size = batch_size
        self.gp_weight = gp_weight
        self.critic_iter = critic_iter
        self.device = device
        self.scorepath = scorepath
        
        self.data_train = data_train
        
        self.G.to(self.device)
        self.D.to(self.device)

        
        self.latent_dim = latent_dim
        self.ts_dim = ts_dim
        self.losses = {'G': [], 'D': [], 'GP': [], 'gradient_norm': [], 'plot_analysis_sf': [], 'plot_analysis_m': []}
        
        
    def train(self, epochs):
        start = time()
        plot_num = 0
        for epoch in range (1, epochs+1):

            for i in range(self.critic_iter):
                #train the critic
                noises = torch.normal(mean = 0, std = 0.3, size =(self.batch_size, self.latent_dim)).to(self.device)
                fake_batch = self.G(noises)
                real_batch = self.data_train[np.random.randint(low = 0, high = self.data_train.cpu().shape[0], size = self.batch_size)]
                
                fake_batch.to(self.device)
                real_batch.to(self.device)
                
                d_fake = self.D(fake_batch)
                d_real = self.D(real_batch)
                
                grad_penalty, grad_norm_ = self.compute_gp(real_batch, fake_batch)
                self.D_opt.zero_grad()
                
                #d_loss computation
                d_loss = d_fake.mean() - d_real.mean() + grad_penalty.to(torch.float32)
                d_loss.backward()
                self.D_opt.step()
                
                if i == self.critic_iter-1:
                    self.losses['D'].append(float(d_loss))
                    self.losses['GP'].append(grad_penalty.item())
                    self.losses['gradient_norm'].append(float(grad_norm_))
                    
                
            
            #train the generator
            self.G_opt.zero_grad()
            noises = torch.normal(mean = 0, std = 0.3, size = (self.batch_size, self.latent_dim)).to(self.device)
            fake_critic_batch = self.G(noises)
                
            fake_critic_batch.to(self.device)
                
            d_critic_fake = self.D(fake_critic_batch)
            
            #g_loss computation  
            g_loss = - d_critic_fake.mean()
            g_loss.backward()
            self.G_opt.step()
            
            #save the loss of feed forward
            self.losses['G'].append(g_loss.item())
            
            #save the data every 1000 epochs
            if epoch % 1000 == 0:
                path = os.path.join(self.scorepath, str(epoch))
                if not os.path.exists(path):
                    os.makedirs(path)
                plt_result_sf = Analysis_sf(self.data_train[: self.batch_size], fake_critic_batch, path)
                self.losses['plot_analysis_sf'].append(plt_result_sf)
                plt_result_m = Analysis_m(self.data_train[: self.batch_size], fake_critic_batch, path)
                self.losses['plot_analysis_m'].append(plt_result_m)
                
                #plot_num = plot_num+1
                
                noises = torch.normal(mean = 0, std = 0.3, size =(self.batch_size, self.latent_dim)).to(self.device)
                torch.save(self.G.state_dict(), path + '\generator' + str(epoch) +' epochs.pth')
                torch.save(self.D.state_dict(), path + '\discriminator' + str(epoch) +' epochs.pth')
                
                #print q sample from the synthetic data 
                y = self.G(noises)
                with torch.no_grad():
                    plt.figure(figsize=(9,4.5))
                    plt.plot(y[0].cpu())
                    plt.title('Simulation after training Epochs = '+str(epoch))
                    plt.savefig(self.scorepath + '\Simulation after training Epochs = '+str(epoch) + '.png', bbox_inches = 'tight', pad_inches = 0.5)
                    plt.close()
                    
                
        end = time()
        
        self.temps = end - start
                
                
    def compute_gp(self, real_data, fake_data):
        batch_size = self.batch_size
        #sample t from uniform distribution
        t = torch.rand(batch_size, self.ts_dim).to(self.device)
        t = t.expand_as(real_data)
        
        #interpolation between real data and fake data
        interpolation = t * real_data + (1-t)*fake_data
        
        interp_logits = self.D(interpolation)
        
        #compute gradient
        torch.autograd.set_detect_anomaly(True)
        gradients = torch_grad(outputs=interp_logits, inputs=interpolation, grad_outputs=torch.ones(interp_logits.size()).to(self.device), create_graph=True, retain_graph=True)[0]
        
        #compute the gradient norm
        gradients = gradients.view(batch_size, -1)
        
        eps = 1e-10
        gradients_norm = torch.sqrt(torch.sum(gradients**2, dim=1, dtype=torch.double) + eps)
        
        return self.gp_weight * (torch.max(torch.zeros(1,dtype=torch.double).to(self.device) ,gradients_norm.mean()-1)**2), gradients_norm.mean().item()