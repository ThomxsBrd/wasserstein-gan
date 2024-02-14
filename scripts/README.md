## Summary of the Python files

1. **main.py**: Performs training for a mrw and displays the associated analysis results.

2. **models.py**: Contains the classes _Funct_ (final activation function), _Generator(kernel_size, use_Dense, latent_dim)_ and _Discriminator(ts_dim)_.

3. **plot_result.py**: Contains the classes _Analysis_sf_ and _Analysis_m_. 
*  _Analysis_sf_ calculates the results of the structure function and the cumulants. For wavelet analysis: _regression_result_dwt_ and _regression_dwt_. For cumulant analysis: _regression_result_lwt_ and _regression_lwt_. Use _.plot()_ and _.dataFrame()_ to display the result.

* _Analysis_m_ calculates mean, std, skewness, and kurtosis for each signal. Use _.dataFrame()_ and _.boxplot()_ to display the result.

4. **training.py**: Contains the class _Trainer_ with two functions _train(n_epochs)_ and _compute_gp(real_data, fake_data)_.

5. **fbm.py**, **mrw.py**, **pzutils.py**: These three files are part of the pymultifracs librairy introduced on the first page

The following piece of codes helped to preprocess data before using them to generate surrogate time series:
```
def create_fbm(latent_dim, dim, H, device):
    data = []
    ecart = []
    for _ in range (dim):
        x = fbm(latent_dim+1, H)
        x = np.diff(x)
                
        data.append(x)
        ecart.append(np.array(x).std())
            
    ecart = np.array(ecart)
    sigma = ecart.mean()
    data = np.array(data) / sigma * 0.3
    data = data.astype('float32')

    return torch.tensor(data).to(device) 
 ```


```
def create_mrw(latent_dim, dim, H, lambda2, L, device):
    data = []
    ecart = []
    for _ in range (dim):
        x = mrw(latent_dim+1, H, np.sqrt(lambda2), L)
        x = np.diff(x)
                
        data.append(x)
        ecart.append(np.array(x).std())
            
    ecart = np.array(ecart)
    sigma = ecart.mean()
    data = np.array(data) / sigma * 0.1
    data = data.astype('float32')

    return torch.tensor(data).to(device) 
 ```
This code allowed us to count the number of parameters in each one of our models:

```
from prettytable import PrettyTable 

def count_parameters(model): 
    table = PrettyTable(["Modules", "Parameters"]) 
    total_params = 0 
    for name, parameter in model.named_parameters(): 
        if not parameter.requires_grad: continue 
        param = parameter.numel() 
        table.add_row([name, param]) 
        total_params+=param 
    print(table) 
    print(f"Total Trainable Params: {total_params}") 
```
