# Deep Learning pour la gestion d'actif
# Vivienne Investissmeent 
#
# Analysis of wavelet coefficients
#
# Thomas Beroud
# Aug, 2022

import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from math import log2
from scipy.stats import skew, kurtosis

from IPython.display import display
import dataframe_image as dfi


from pymultifracs.wavelet import wavelet_analysis
import pymultifracs.mfa as mf


    #
    # Structure Function Analysis + C1 + C2
    #
    
    
class Analysis_sf():  
    def __init__(self, data_train, synthetic_data, path):
        self.data_train = data_train.cpu()
        self.synthetic_data = synthetic_data.cpu()
        self.path = path
        
        dwt, lwt = mf.mf_analysis_full(self.data_train[0].detach().numpy(), j1=3, j2=int(log2(len(self.data_train[0])) - 5), q=[2], n_cumul=3, gamint=0.0)
        sf, cumul, mfs, hmin = dwt
        
        self.j = sf.j
        
        #analyse wavelet
        self.wavelet_coef_data = self.regression_result_dwt(self.data_train) #return all wavelet coefs for all signals 
        self.wavelet_coef_sync =self.regression_result_dwt(self.synthetic_data)
        
        
        self.wavelet_result_data, self.wavelet_slope_data, self.wavelet_std_slope_data = self.regression_dwt(self.wavelet_coef_data)
        self.wavelet_result_synthetic, self.wavelet_slope_synthetic, self.wavelet_std_slope_synthetic = self.regression_dwt(self.wavelet_coef_sync)
        
        
        #analyse cumulant
        self.cumul_coef_data = self.regression_result_lwt(self.data_train)
        self.cumul_coef_sync = self.regression_result_lwt(self.synthetic_data)
        
        self.cumul_result1_data, self.cumul_result2_data, self.cumul_hmin_data, self.cumul_slope_data, self.cumul_std_slope_data = self.regression_lwt(self.cumul_coef_data)
        self.cumul_result1_synthetic, self.cumul_result2_synthetic, self.cumul_hmin_synthetic, self.cumul_slope_synthetic, self.cumul_std_slope_synthetic = self.regression_lwt(self.cumul_coef_sync)
        
    
    
    #analyse wavelet
    def regression_result_dwt(self, data):
        
        with torch.no_grad():
            #prevent case data is on cuda
            data = data.cpu()
            wavelet_coef  = {'slope' :[]}
            
            for i in range (len(data)):
                dwt, lwt = mf.mf_analysis_full(data[i].detach().numpy(), j1=3, j2=int(log2(len(data[i])) - 5), q=[2], n_cumul=3, gamint=0.0)
                sf, cumul, mfs, hmin = dwt
                
                if i == 0:
                    for k in range (len(self.j)):
                        wavelet_coef.update({'j' + str(k+1) : []})
            
                for j in range (len(self.j)):
                    wavelet_coef['j'+str(j+1)].append(sf.logvalues[0,j,0])
                wavelet_coef['slope'].append(sf.zeta[0])
        
        return wavelet_coef
    
    
    #analyse cumulants
    def regression_result_lwt(self, data):
        
        with torch.no_grad():
            #prevent case data is on cuda
            data = data.cpu()
            cumul_coef  = {'hmin' :[], 'slope1' :[], 'slope2' :[]}
            
            for i in range (len(data)):
                dwt, lwt = mf.mf_analysis_full(data[i].detach().numpy(), j1=3, j2=int(log2(len(data[i])) - 5), q = [-2,1,0,1,2], n_cumul=2, p_exp=np.inf, gamint=1.)
                lwt_sf, lwt_cumul, lwt_mfs, hmin = lwt
                
                if i == 0:
                    for k in range (len(self.j)):
                        cumul_coef.update({'m1j' + str(k+1) : []})
                        cumul_coef.update({'m2j' + str(k+1) : []})
            
                for k in range(len(lwt_cumul.log_cumulants)):
                    for j in range (len(self.j)):
                        cumul_coef['m'+str(k+1) +'j'+str(j+1)].append(lwt_cumul.values[k][j])
                    
                cumul_coef['hmin'].append(hmin)
                cumul_coef['slope1'].append(lwt_cumul.log_cumulants[0])
                cumul_coef['slope2'].append(lwt_cumul.log_cumulants[1])
        return cumul_coef
    
    
    def regression_dwt(self, wavelet_coef):
        
        mean_wavelet_coef = []
        std_wavelet_coef = []
        skew_wavelet_coef = []
        kurtosis_wavelet_coef = []
        
        for i in range (len(self.j)):
            mean_wavelet_coef.append(np.array(wavelet_coef['j'+str(i+1)]).mean())
            std_wavelet_coef.append(np.array(wavelet_coef['j'+str(i+1)]).std())
            skew_wavelet_coef.append(skew(np.array(wavelet_coef['j'+str(i+1)])))
            kurtosis_wavelet_coef.append(kurtosis(np.array(wavelet_coef['j'+str(i+1)])))
            
        result = [np.array(mean_wavelet_coef), np.array(std_wavelet_coef), np.array(skew_wavelet_coef), np.array(kurtosis_wavelet_coef)]
                
        mean_slope = np.array(wavelet_coef['slope']).mean()
        std_slope = np.array(wavelet_coef['slope']).std()
            
        return [result, mean_slope, std_slope]
    
    def regression_lwt(self, cumul_coef):
        
        mean_cumul_m1 = []
        std_cumul_m1 = []
        skew_cumul_m1 = []
        kurtosis_cumul_m1 = []
        
        mean_cumul_m2 = []
        std_cumul_m2 = []
        skew_cumul_m2 = []
        kurtosis_cumul_m2 = []
            
        for i in range (len(self.j)):
            mean_cumul_m1.append(np.array(cumul_coef['m1j'+str(i+1)]).mean())
            std_cumul_m1.append(np.array(cumul_coef['m1j'+str(i+1)]).std())
            skew_cumul_m1.append(skew(np.array(cumul_coef['m1j'+str(i+1)])))
            kurtosis_cumul_m1.append(kurtosis(np.array(cumul_coef['m1j'+str(i+1)])))
            
            mean_cumul_m2.append(np.array(cumul_coef['m2j'+str(i+1)]).mean())
            std_cumul_m2.append(np.array(cumul_coef['m2j'+str(i+1)]).std())
            skew_cumul_m2.append(skew(np.array(cumul_coef['m2j'+str(i+1)])))
            kurtosis_cumul_m2.append(kurtosis(np.array(cumul_coef['m2j'+str(i+1)])))
            
            
        result_m1 = [np.array(mean_cumul_m1), np.array(std_cumul_m1), np.array(skew_cumul_m1), np.array(kurtosis_cumul_m1)]
        result_m2 = [np.array(mean_cumul_m2), np.array(std_cumul_m2), np.array(skew_cumul_m2), np.array(kurtosis_cumul_m2)]
        
        mean_hmin = [np.array(cumul_coef['hmin']).mean(), np.array(cumul_coef['hmin']).std()]
        
        mean_slope = [np.array(cumul_coef['slope1']).mean(), np.array(cumul_coef['slope2']).mean()]
        std_slope = [np.array(cumul_coef['slope1']).std(), np.array(cumul_coef['slope2']).std()]
            
        return [result_m1, result_m2, mean_hmin, mean_slope, std_slope]


    
    def plot(self):
        
        plt.figure(1,figsize=(35,8))
        #plt.gcf().subplots_adjust(left = 0.2, bottom = 0.2, right = 1.5, top = 0.8, wspace = 0.5, hspace = 0.5)
        #fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (35,5))

        #plot mean structure fonction from data_train
        plt.subplot(2,2,1)
        plt.plot(self.j, self.wavelet_result_data[0],'--', c='r', label = 'Training Set, Slope = ' +'%.5f' % (self.wavelet_slope_data))
        plt.scatter(self.j, self.wavelet_result_data[0], c='r',s=10)
        plt.errorbar(self.j, self.wavelet_result_data[0], yerr = self.wavelet_result_data[1], fmt = 'none', capsize = 10, ecolor = 'r', elinewidth = 1, capthick = 2)
    
        #plot mean structure fonction from data_gererated 
        plt.plot(self.j, self.wavelet_result_synthetic[0],'-', c='b', label = 'Synthetic Data, Slope = ' +'%.5f' % (self.wavelet_slope_synthetic)+ ' - Std Slope = ' +'%.5f' % (self.wavelet_std_slope_synthetic))
        plt.scatter(self.j, self.wavelet_result_synthetic[0], c='b',s=10)
        plt.errorbar(self.j, self.wavelet_result_synthetic[0], yerr = self.wavelet_result_synthetic[1], fmt = 'none', capsize = 10, ecolor = 'b', elinewidth = 1, capthick = 2)
         
        plt.xlabel('j')
        plt.title('Wavelet Coef - Mean Structure Function $log_2(S(j))$, $q=2$')
        plt.legend()
        
        #plot the mean structure fonction from data_gererated minus the true mean structure fonction from data_train
        plt.subplot(2,2,3)
        X = np.zeros(len(self.j))
        plt.plot(self.j, X,'--', c='r')
        plt.scatter(self.j, X, c='r',s=10)
        plt.errorbar(self.j, X, yerr = self.wavelet_result_data[1], fmt = 'none', capsize = 10, ecolor = 'r', elinewidth = 1, capthick = 2)
        
        X = self.wavelet_result_synthetic[0] - self.wavelet_result_data[0]
    
        #plot mean structure fonction from data_gererated 
        plt.plot(self.j, X,'-', c='b')
        plt.scatter(self.j, X, c='b',s=10)
        plt.errorbar(self.j, X, yerr = self.wavelet_result_synthetic[1], fmt = 'none', capsize = 10, ecolor = 'b', elinewidth = 1, capthick = 2)
    
        plt.xlabel('j')
        
        plt.subplot(2,2,2)
        plt.plot(self.j, self.cumul_result1_data[0],'--', c='r', label = 'Training Set, [$ x \log_2(e)]$ = ' +'%.5f' % (self.cumul_slope_data[0]))
        plt.scatter(self.j, self.cumul_result1_data[0], c='r',s=10)
        plt.errorbar(self.j, self.cumul_result1_data[0], yerr = self.cumul_result1_data[1], fmt = 'none', capsize = 10, ecolor = 'r', elinewidth = 1, capthick = 2)
    
        #plot mean structure fonction from data_gererated 
        plt.plot(self.j, self.cumul_result1_synthetic[0],'-', c='b', label = 'Synthetic Data, [$ x \log_2(e)]$ = ' +'%.5f' % (self.cumul_slope_synthetic[0])+ ' - Std = ' +'%.5f' % (self.cumul_std_slope_synthetic)[0])
        plt.scatter(self.j, self.cumul_result1_synthetic[0], c='b',s=10)
        plt.errorbar(self.j, self.cumul_result1_synthetic[0], yerr = self.cumul_result1_synthetic[1], fmt = 'none', capsize = 10, ecolor = 'b', elinewidth = 1, capthick = 2)
        
        plt.title ('Wavelet Leader - Cumulants $C_m (j)$ ')
        plt.xlabel('j')
        plt.ylabel('m = 1')
        plt.legend()
        
        plt.subplot(2,2,4)
        plt.plot(self.j, self.cumul_result2_data[0],'--', c='r', label = 'Training Set, [$ x \log_2(e)]$ = ' +'%.5f' % (self.cumul_slope_data[1]))
        plt.scatter(self.j, self.cumul_result2_data[0], c='r',s=10)
        plt.errorbar(self.j, self.cumul_result2_data[0], yerr = self.cumul_result2_data[1], fmt = 'none', capsize = 10, ecolor = 'r', elinewidth = 1, capthick = 2)
    
        #plot mean structure fonction from data_gererated 
        plt.plot(self.j, self.cumul_result2_synthetic[0],'-', c='b', label = 'Synthetic Data, [$ x \log_2(e)]$ = ' +'%.5f' % (self.cumul_slope_synthetic[1])+ ' - Std = ' +'%.5f' % (self.cumul_std_slope_synthetic[1]))
        plt.scatter(self.j, self.cumul_result2_synthetic[0], c='b',s=10)
        plt.errorbar(self.j, self.cumul_result2_synthetic[0], yerr = self.cumul_result2_synthetic[1], fmt = 'none', capsize = 10, ecolor = 'b', elinewidth = 1, capthick = 2)
        plt.xlabel('j')
        plt.ylabel('m = 2')
        plt.legend()
        plt.savefig(self.path + '\Mean Structure Function Analysis.png', bbox_inches = 'tight', pad_inches = 0.5)
        plt.show()

        
        
        #plt.legend()
        #plt.savefig(self.path + '\Wavelet Coef - Mean Structure Function Analysis.png', bbox_inches = 'tight', pad_inches = 0.5)
        #plt.draw()

        
    def dataFrame(self):
    
        ar_data = np.array([self.wavelet_result_data[0], self.wavelet_result_data[1], self.wavelet_result_data[2], self.wavelet_result_data[3]])
        df_data = pd.DataFrame(ar_data, index = ['Mean', 'Std', 'Skewness', 'Kurtosis'], columns = pd.MultiIndex.from_product([['Training Set'],['j'+str(i+1) for i in range (len(self.j))]], names=['Data:', 'j:']))
        
        df_data = df_data.style.format('{:.4f}')
        df_data = df_data.set_table_styles([{'selector': '.index_name', 'props': 'font-style: italic; color: darkgrey; font-weight:normal;'},{'selector': 'th:not(.index_name)', 'props': 'background-color: #000066; color: white; text-align: center;'},{'selector': 'tr:hover', 'props': [('background-color', '#ffffb3')]},], overwrite=False)

        ar_synthetic = np.array([self.wavelet_result_synthetic[0], self.wavelet_result_synthetic[1], self.wavelet_result_synthetic[2], self.wavelet_result_synthetic[3]])
        df_synthetic = pd.DataFrame(ar_synthetic, index = ['Mean', 'Std', 'Skewness', 'Kurtosis'], columns = pd.MultiIndex.from_product([['Synthetic Data'],['j'+str(i+1) for i in range (len(self.j))]], names=['Data:', 'j:']))
        df_synthetic = df_synthetic.style.format('{:.4f}')
        df_synthetic = df_synthetic.set_table_styles([{'selector': '.index_name', 'props': 'font-style: italic; color: darkgrey; font-weight:normal;'},{'selector': 'th:not(.index_name)', 'props': 'background-color: #000066; color: white; text-align: center;'},{'selector': 'tr:hover', 'props': [('background-color', '#ffffb3')]},], overwrite=False)

        display(df_data)
        display(df_synthetic)
        dfi.export(df_data, self.path + '\DataFrame Wavelet Coef Training Set.png')
        dfi.export(df_synthetic, self.path + '\DataFrame Wavelet Coef Synthetic Data.png')
        
        return df_data, df_synthetic 
    
        
    #
    # Moments, Skewness & Kurtosis Analysis 
    #
        
class Analysis_m():
    
    def __init__(self, data_train, synthetic_data, path):

        self.data_train = data_train.cpu()
        self.synthetic_data = synthetic_data.cpu()
        self.path = path  #path for saving images
        self.moment_data_train = self.compute_moments(self.data_train)
        self.moment_synthetic_data = self.compute_moments(self.synthetic_data)

        
    def compute_moments(self,data):
        with torch.no_grad():
            #prevent case data is on cuda
            data = data.cpu()
            moment  = {'mean': [], 'std': [], 'skew': [], 'kurtosis': []}
            for i in range (len(data)):
                moment['mean'].append(data[i].mean())
                moment['std'].append(data[i].std())
                moment['skew'].append(skew(data[i]))
                moment['kurtosis'].append(kurtosis(data[i]))
        return moment

    #print and save dataFrame
    def dataFrame(self):
        moment = self.moment_data_train
        mean_train = np.array(moment['mean']).mean()
        std_train = np.array(moment['std']).mean()
        skew_train = np.array(moment['skew']).mean()
        kurtosis_train = np.array(moment['kurtosis']).mean()
        
        moment = self.moment_synthetic_data
        mean_synthetic = np.array(moment['mean']).mean()
        std_synthetic = np.array(moment['std']).mean()
        skew_synthetic = np.array(moment['skew']).mean()
        kurtosis_synthetic = np.array(moment['kurtosis']).mean()
    
        ar = np.array([[mean_train, mean_synthetic], [std_train, std_synthetic], [skew_train, skew_synthetic], [kurtosis_train, kurtosis_synthetic]])
        
        df = pd.DataFrame(ar, index = ['Mean', 'Std', 'Skewness', 'Kurtosis'], columns = ['Training Set', 'Synthetic Data'])
        df = df.style.format('{:.5f}')
        df = df.set_table_styles([{'selector': '.index_name', 'props': 'font-style: italic; color: darkgrey; font-weight:normal;'},{'selector': 'th:not(.index_name)', 'props': 'background-color: #000066; color: white; text-align: center;'},{'selector': 'tr:hover', 'props': [('background-color', '#ffffb3')]},], overwrite=False)
        
        display(df)
        dfi.export(df, self.path + '\Mean Moments Data.png')
        
        return df

    #print and save boxplot
    def boxplot(self):
    
        fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (10, 4))
        axes[0].boxplot([self.moment_data_train['skew'],self.moment_synthetic_data['skew']], labels = ['Training Set','Synthetic Data'] )
        axes[0].set_title('Skewness boxplot of real and synthetic series')
    
        axes[1].boxplot([self.moment_data_train['kurtosis'],self.moment_synthetic_data['kurtosis']], labels = ['Training Set','Synthetic Data'] )
        axes[1].set_title('Kurtosis boxplot of real and synthetic series')
        
        fig.tight_layout()
        plt.savefig(self.path + '\Kurtosis & Kurtosis boxplot of real and synthetic series.png', bbox_inches = 'tight', pad_inches = 0.5)
        plt.show()