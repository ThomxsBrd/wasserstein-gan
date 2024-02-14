# Deep Learning pour la gestion d'actif
# Vivienne Investissmeent 
#
# Generator & Critic
#
# Thomas Beroud
# Aug, 2022


# Import of the libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

# Final Activation funtion for the Generator 

class Funct(nn.Module):
    
    def __init__(self, alpha = None, beta = None, initialisation = 0.5):
        super(Funct,self).__init__()
        
        self.tanh = nn.Tanh()
        
        if alpha == None :
            self.alpha = Parameter(torch.tensor(initialisation))
        else :
            self.alpha = Parameter(torch.tensor(alpha))
        self.alpha.requiresGrad = True
        
        if beta == None :
            self.beta = Parameter(torch.tensor(initialisation))
        else :
            self.beta = Parameter(torch.tensor(alpha))
        self.beta.requiresGrad = True
        
    def forward(self,x): 
        return self.beta* self.tanh(self.alpha * x)
    
    
# Generator

class Generator(nn.Module):
    def __init__(self, kernel_size = 32, use_Dense = False, latent_dim = 2**12):
        super(Generator, self).__init__()
        
        if use_Dense == False:
            self.Layer0 = nn.Conv1d(1, 1, kernel_size = kernel_size, padding = 'same')
        else :
            self.Layer0 = nn.Linear(latent_dim,latent_dim)
            
        self.batch0 = nn.BatchNorm1d(1)
        
        self.conv1 = nn.ConvTranspose1d(8, 4, kernel_size = 7, stride = 2, output_padding = 1, padding = 3)
        self.batch1 = nn.BatchNorm1d(4)
        
        self.conv2 = nn.ConvTranspose1d(4, 2, kernel_size = 7, stride = 2, output_padding = 1, padding = 3)
        self.batch2 = nn.BatchNorm1d(2)
        
        self.conv3 = nn.ConvTranspose1d(2, 1, kernel_size = 7, stride = 2, output_padding = 1, padding = 3)
        self.batch3 = nn.BatchNorm1d(1)
        
        #final activation function 
        self.f = Funct()


    
    def forward(self, x):
        x = x.view(x.shape[0], 1, x.shape[1])
        
        x = self.Layer0(x)
        x = self.batch0(x)
        x = F.leaky_relu(x, 0.2)
        
        #Reshaping
        x = x.view(x.shape[0], 8, x.shape[2]//8)
        
        x = self.batch1(self.conv1(x))
        x = F.leaky_relu(x, 0.2)
        
        x = self.batch2(self.conv2(x))
        x = F.leaky_relu(x, 0.2)
        
        x = self.batch3(self.conv3(x))
        
        x = x.squeeze()
        
        return self.f(x)
    
# Critic / Discriminator

class Discriminator(nn.Module):
    def __init__(self, ts_dim):
        super(Discriminator, self).__init__()
        self.ts_dim = ts_dim
        
        self.conv1 = nn.Conv1d(1, 32, kernel_size = 7, padding = 'same')
        self.conv2 = nn.Conv1d(32, 32, kernel_size = 7, padding = 'same')
        self.conv3 = nn.Conv1d(32, 32, kernel_size = 7, padding = 'same') 
        
        self.flatten = nn.Flatten()
        
        self.dense1 = nn.Linear(self.ts_dim*8, 50) 
        self.dense2 = nn.Linear(50, 15)
        self.dense3 = nn.Linear(15, 1)    
        
    def forward(self, x):
        x = x.view(x.shape[0], 1, x.shape[1])
        
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.max_pool1d(x, 2)
        
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.max_pool1d(x, 2)
        
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.max_pool1d(x,1)
        
        x = self.flatten(x)
        x = F.leaky_relu(self.dense1(x), 0.2)
        x = F.leaky_relu(self.dense2(x), 0.2)
        return F.leaky_relu(self.dense3(x), 0.2)