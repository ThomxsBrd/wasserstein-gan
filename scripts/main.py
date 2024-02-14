# Deep Learning pour la gestion d'actif
# Vivienne Investissmeent 
#
# Main
#
# Thomas Beroud
# Aug, 2022

# Init librairies

import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os 
from scipy.io import savemat
import torch.optim as optim
from torch.autograd import grad as torch_grad
import seaborn as sns
sns.set(style="darkgrid")

from models import Generator, Discriminator
from training import Trainer
from plot_result import Analysis_sf, Analysis_m

# Init parameters
latent_dim = 2**12
ts_dim = 2**12
n_epochs = 1
batch_size = 128
device = 'cuda' if torch.cuda.is_available() else 'cpu'

lr_a = 1e-4
lr_b = 1e-4

kernel_size = 32
use_Dense = False

score_data_train = r'C:\Users\Stage\Desktop\WGAN_times_series\data_train\training_set_mrw_0.8_0.03.csv'
true_data = torch.tensor(pd.read_csv(score_data_train).to_numpy().astype('float32')).to(device)
data_train = true_data[np.random.randint(low = 0, high = true_data.cpu().shape[0], size = 10000)] 

# Create Scorepath
name = 'try'
scorepath = os.path.join(r'C:\Users\Stage\Desktop\WGAN_times_series\Synthetization of a fbm\output',name)
if not os.path.exists(scorepath):
    os.makedirs(scorepath)

# Init models
generator = Generator(kernel_size, use_Dense, latent_dim).to(device)
discriminator = Discriminator(ts_dim = ts_dim).to(device)
        
# Init optimizers
G_opt = optim.RMSprop(generator.parameters(), lr=lr_a)
D_opt = optim.RMSprop(discriminator.parameters(), lr=lr_b)

# Training and Saving models
train = Trainer(generator, discriminator, G_opt, D_opt, batch_size, latent_dim, ts_dim, data_train, device, scorepath)
train.train(epochs=n_epochs)

#Print Synthetic Time Series
noise = torch.normal(mean = 0, std = 0.3, size = (10000, latent_dim)).to(device)
synthetic_data = generator(noise).cpu()
with torch.no_grad():
    x = synthetic_data[np.random.randint(low = 0, high = 10000)]
    plt.plot(x, linewidth = 0.5)
    plt.title('Realization')
    plt.show()

# Print
analysis_m = Analysis_m(data_train, synthetic_data, scorepath)
analysis_sf = Analysis_sf(data_train, synthetic_data, scorepath)
analysis_m.dataFrame()
analysis_m.boxplot()
analysis_sf.plot()
df_data, df_synthetic = analysis_sf.dataFrame()


def leverage(r):
    R, N = r.shape
    #dr = np.diff(r, axis=1)
    r2 = r ** 2
    L = np.stack([np.correlate(r2[k], r[k], mode='full') / N  for k in range(R)], axis=0)
    L = np.mean(L, axis=0) / (np.mean(r2))
    tau = np.arange(-N+1, N)
    return L, tau

lev, tau = leverage(data_train.cpu().detach().numpy())
lev_syn, t = leverage(synthetic_data.cpu().detach().numpy())

plt.plot(tau, lev, c = 'r')
plt.plot(tau, lev_syn, c = 'b')
plt.xlim(-10, 50)
plt.ylim(-0.03,0.03)
plt.xlabel('tau')
plt.ylabel('L(tau)')
plt.title('Leverage correlation')
plt.tight_layout()
plt.show()