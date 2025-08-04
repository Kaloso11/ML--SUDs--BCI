#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 15:36:33 2023

@author: kaloso
"""


import scipy
import numpy as np
import pandas as pd
from scipy import stats, signal, integrate
from dit.other import tsallis_entropy
import dit
import librosa
import statsmodels.api as sm
import itertools
from pyinform import mutualinfo
import statsmodels.api as sm
from statsmodels.tsa import stattools
#import statsmodels.api as sm
from sklearn.metrics import mutual_info_score
from scipy import signal,integrate
from scipy.signal import coherence
from scipy.signal import hilbert,periodogram, butter, lfilter
from scipy.integrate import simps
from sklearn.metrics.cluster import normalized_mutual_info_score as normed_mutual_info 
# import numba
from numba import jit, prange
# import torch
import librosa
from multiprocessing import Pool



################################################
#	Connectivity features
################################################


def compute_band_coherence_manual(eegData, i, j, epoch, fs=128):
    # Compute the FFT for the signals of a specific epoch
    X = np.fft.fft(eegData[i, :, epoch])
    Y = np.fft.fft(eegData[j, :, epoch])
    
    # Compute cross-spectral density and power spectral densities
    Sxy = X * np.conj(Y)
    Sxx = X * np.conj(X)
    Syy = Y * np.conj(Y)

    # Compute coherence
    C = np.abs(Sxy) / np.sqrt(np.abs(Sxx) * np.abs(Syy))
    
    # Frequency vector
    f = np.fft.fftfreq(eegData.shape[1], 1/fs)
    
    # Compute mean coherence for each frequency band
    delta_coherence = np.mean(C[(f >= 0.5) & (f <= 4)])
    theta_coherence = np.mean(C[(f >= 4) & (f <= 8)])
    alpha_coherence = np.mean(C[(f >= 8) & (f <= 12)])
    beta_coherence = np.mean(C[(f >= 12) & (f <= 25)])
    gamma_coherence = np.mean(C[(f >= 25) & (f <= 55)])

    return delta_coherence, theta_coherence, alpha_coherence, beta_coherence, gamma_coherence

def coherence(eegData, fs):
    n_channels, _, n_epochs = eegData.shape
    coherences = {'delta': np.zeros((n_channels, n_epochs)),
                  'theta': np.zeros((n_channels, n_epochs)),
                  'alpha': np.zeros((n_channels, n_epochs)),
                  'beta': np.zeros((n_channels, n_epochs)),
                  'gamma': np.zeros((n_channels, n_epochs))}
    
    for epoch in range(n_epochs):
        all_coherences = {'delta': [], 'theta': [], 'alpha': [], 'beta': [], 'gamma': []}
        
        for ii, jj in itertools.combinations(range(n_channels), 2):
            delta, theta, alpha, beta, gamma = compute_band_coherence_manual(eegData, ii, jj, epoch, fs=fs)
            
            all_coherences['delta'].append(delta)
            all_coherences['theta'].append(theta)
            all_coherences['alpha'].append(alpha)
            all_coherences['beta'].append(beta)
            all_coherences['gamma'].append(gamma)
        
        # Average coherence for each band and each epoch
        for key in coherences:
            coherences[key][:, epoch] = np.mean(all_coherences[key])

    return coherences



##########
# Mutual information
def calculate2Chan_MI(eegData, ii, jj, bin_min=-200, bin_max=200, binWidth=2):
    H = np.zeros(eegData.shape[2])
    bins = np.arange(bin_min+1, bin_max, binWidth)
    for epoch in range(eegData.shape[2]):
        c_xy, _, _ = np.histogram2d(eegData[ii, :, epoch], eegData[jj, :, epoch], [bins, bins])
        H[epoch] = mutual_info_score(None, None, contingency=c_xy)
    return H

def calculate_all_pairs_MI(eegData):
    n_channels = eegData.shape[0]
    epochs = eegData.shape[2]
    
    # Initialize the result array: One row for each channel
    all_MI_array = np.zeros((n_channels, epochs))

    for ii in range(n_channels):
        # For each channel ii, compute the average MI with all other channels
        total_MI_for_channel = np.zeros(epochs)
        for jj in range(n_channels):
            if ii != jj:
                MI_values = calculate2Chan_MI(eegData, ii, jj)
                total_MI_for_channel += MI_values
        # Average the mutual information over all pairs involving channel ii
        all_MI_array[ii, :] = total_MI_for_channel / (n_channels - 1)

    return all_MI_array

# ##########
# # Granger causality
# # Define the calcGrangerCausality function based on your provided code
# def calcGrangerCausality(eegData, ii, jj):
#     H = np.zeros(eegData.shape[2])
#     for epoch in range(eegData.shape[2]):
#         X = np.vstack([eegData[ii, :, epoch], eegData[jj, :, epoch]]).T
#         H[epoch] = stattools.grangercausalitytests(X, 1, addconst=True, verbose=False)[1][0]['ssr_ftest'][0]
#     return H

# # Define the function to calculate pairwise Granger causality
# def calc_pairwise_granger(args):
#     eegData, ii, jj = args
#     return calcGrangerCausality(eegData, ii, jj)

# # Define the function to compute all Granger causality in parallel
# def compute_all_granger_parallel(eegData):
#     n_channels = eegData.shape[0]
#     n_epochs = eegData.shape[2]
#     GC_matrix = np.zeros((n_channels, n_channels, n_epochs))
    
#     with Pool() as pool:
#         results = pool.map(calc_pairwise_granger, [(eegData, ii, jj) for ii in range(n_channels) for jj in range(n_channels) if ii != jj])
    
#     index = 0
#     for ii in range(n_channels):
#         for jj in range(n_channels):
#             if ii != jj:
#                 GC_matrix[ii, jj, :] = results[index]
#                 index += 1
                
#     return GC_matrix



##########
# # phase Lag Index
# def phaseLagIndex(eegData, i, j):
#     hxi = hilbert(eegData[i,:,:])
#     hxj = hilbert(eegData[j,:,:])
#     # calculating the INSTANTANEOUS PHASE
#     inst_phasei = np.arctan(np.angle(hxi))
#     inst_phasej = np.arctan(np.angle(hxj))

#     out = np.abs(np.mean(np.sign(inst_phasej - inst_phasei), axis=0))
#     return out

##########
# # Cross-correlation Magnitude
# def crossCorrMag(eegData,ii,jj):
# 	crossCorr_res = []
# 	for ii, jj in itertools.combinations(range(eegData.shape[0]), 2):
# 		crossCorr_res.append(crossCorrelation(eegData, ii, jj))
# 	crossCorr_res = np.array(crossCorr_res)
# 	return crossCorr_res

# ##########
# # Auxilary Cross-correlation Lag
# def corrCorrLagAux(eegData,ii,jj,Fs=100):
#     out = np.zeros(eegData.shape[2])
#     lagCorr = []
#     for lag in range(0,eegData.shape[1],int(0.2*Fs)):
#         tmp = eegData.copy()
#         tmp[jj,:,:] = np.roll(tmp[jj,:,:], lag, axis=0)
#         lagCorr.append(cross_correlation(tmp, ii, jj, Fs))
#     return np.argmax(lagCorr,axis=0)

