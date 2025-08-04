#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 14:01:32 2023

@author: kaloso
"""

import EEGExtract_Complex as eegcomp
import numpy as np
#import numba
#from numba import cuda

import warnings
#mne.set_log_level("WARNING")
warnings.filterwarnings("ignore")


# %% Load data

eegData = np.load('batched_data1.npy')
#eegData = cuda.to_device(eeg_data)
fs = 128

# %% #FEATURE EXTRACTION

################################################
#	Complexity features
################################################ 


#Shannon Entropy
ShannonRes = eegcomp.ShannonEntropy(eegData, bin_min=-200, bin_max=200, binWidth=2,fs=fs)

#Tsalis Entropy (n=10)
#talis = eegcomp.tsalisEntropy(eegData, bin_min=-200, bin_max=200, binWidth=2, orders=[1])

# Subband Information Quantity
# delta (0.5–4 Hz)
#eegData_delta = eegcomp.filt_data(eegData, 0.5, 4, fs)
ShannonRes_delta = eegcomp.shannonEntropy(eegData, bin_min=-200, bin_max=200, binWidth=2, lowcut=0.5, highcut=4,fs=fs)
# theta (4–8 Hz)
#eegData_theta = eegcomp.filt_data(eegData, 4, 8, fs)
ShannonRes_theta = eegcomp.shannonEntropy(eegData, bin_min=-200, bin_max=200, binWidth=2,lowcut=4, highcut=8,fs=fs)
# alpha (8–12 Hz)
#eegData_alpha = eegcomp.filt_data(eegData, 8, 12, fs)
ShannonRes_alpha = eegcomp.shannonEntropy(eegData, bin_min=-200, bin_max=200, binWidth=2,lowcut=8, highcut=12,fs=fs)
# beta (12–30 Hz)
#eegData_beta = eegcomp.filt_data(eegData, 12, 25, fs)
ShannonRes_beta = eegcomp.shannonEntropy(eegData, bin_min=-200, bin_max=200, binWidth=2,lowcut=12, highcut=25,fs=fs)
# gamma (30–100 Hz)
#eegData_gamma = eegcomp.filt_data(eegData, 25, 55, fs)
ShannonRes_gamma = eegcomp.shannonEntropy(eegData, bin_min=-200, bin_max=200, binWidth=2,lowcut=25, highcut=55,fs=fs)

# # Cepstrum Coefficients (n=2)
# CepstrumRes = eegcomp.mfcc(eegData, fs,order=2)

# Lyapunov Exponent
lyapunov_res = eegcomp.lyapunov(eegData)

# Fractal Embedding Dimension
HiguchiFD_Res  = eegcomp.hFD(eegData,k_max=10)

# Hjorth Mobilit  ,  # Hjorth Complexity
HjorthMob, HjorthComp = eegcomp.hjorthParameters(eegData)

# False Nearest Neighbor
FalseNnRes = eegcomp.falseNearestNeighbor(eegData)

# ARMA Coefficients (n=2)
#armaRes = eegcomp.arma(eegData,orders=(2,1))



#%% Store the results

feature_list = []
feature_list.append(ShannonRes)
feature_list.append(ShannonRes_delta)
feature_list.append(ShannonRes_theta)
feature_list.append(ShannonRes_alpha)
feature_list.append(ShannonRes_beta)
feature_list.append(ShannonRes_gamma)
feature_list.append(lyapunov_res)
#feature_list.append(HiguchiFD_Res)
feature_list.append(HjorthMob)
feature_list.append(HjorthComp)
feature_list.append(FalseNnRes)


complex_features = np.vstack(feature_list).transpose()


if __name__ == "__main__":
    
    np.save('complex_features1.npy', complex_features)
    
#%%    
