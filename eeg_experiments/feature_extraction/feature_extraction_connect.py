#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 14:01:32 2023

@author: kaloso
"""

import EEGExtract_Connect as eegconn
import numpy as np

import warnings
#mne.set_log_level("WARNING")
warnings.filterwarnings("ignore")

# %% Load data

eegData = np.load('batched_data1.npy')
#eegData = cuda.to_device(eeg_data)
fs = 128

# %% Feature extraction

# ################################################
# #	Connectivity features
# ################################################


coherences = eegconn.coherence(eegData,fs)


# To access coherence values for each band:
coherence_delta = coherences['delta']
coherence_theta = coherences['theta']
coherence_alpha = coherences['alpha']
coherence_beta = coherences['beta']
coherence_gamma = coherences['gamma']

# Mutual Information
info_mu = eegconn.calculate_all_pairs_MI(eegData)

# Granger causality - All
# caus_mes =  eegconn.compute_all_granger_parallel(eegData)

# # Phase Lag Index
# phase = eegconn.PhaseLagIndex(eegData, i, j)

# # Maximum correlation between two signals
# cross_corr = eegconn.crossCorrMag(eegData, ii, jj)

# # time-delay that maximizes correlation between signals
# cross_lag = eegconn.corrCorrLagAux(eegData, ii, jj)


# %% Store the results

feature_list = []
feature_list.append(coherence_delta)
feature_list.append(coherence_theta)
feature_list.append(coherence_alpha)
feature_list.append(coherence_beta)
feature_list.append(coherence_gamma)
feature_list.append(info_mu)
# feature_list.append(caus_mes)
# feature_list.append(ShannonRes_alpha)
# feature_list.append(ShannonRes_beta)
# feature_list.append(ShannonRes_gamma)
# feature_list.append(lyapunov_res)
# #feature_list.append(HiguchiFD_Res)
# feature_list.append(HjorthMob)
# feature_list.append(HjorthComp)
# feature_list.append(FalseNnRes)


connect_features = np.vstack(feature_list).transpose()


if __name__ == "__main__":
    
    np.save('connect_features1.npy', connect_features)
