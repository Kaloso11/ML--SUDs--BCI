#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 15:49:29 2023

@author: kaloso
"""

import bisect
import scipy
import numpy as np
import pandas as pd
import pywt
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
#from numba import jit, njit, cuda
import torch
import librosa



################################################
#	Auxiliary Functions
################################################

##########
# Filter the eegData, midpass filter 
#	eegData: 3D nrray [chans x ms x epochs] 
def filt_data(eegData, lowcut, highcut, fs, order=7):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    filt_eegData = signal.lfilter(b, a, eegData, axis = 1)
    return filt_eegData

#########
# remove short bursts / spikes 
def fcnRemoveShortEvents(z,n):
    for chan in range(z.shape[0]):
        # check for too-short suppressions
        ct=0
        i0=1
        i1=1 
        for i in range(2,len(z[chan,:])):
            if z[chan,i]==z[chan,i-1]:
                ct=ct+1
                i1=i
            else:
                if ct<n:
                    z[chan,i0:i1] = 0
                    z[chan,i1] = 0 #nasty little bug
                ct=0
                i0=i
                i1=i
        if z[chan,0] == 1 and z[chan,1] == 0:
            z[chan,0] = 0
    return z


def get_intervals(A,B,endIdx=500):
    # This function gives you intervals (a1,b1), (a2,b3) for every a in A=[a1,a2,a3,..]
    # and the smallest element in b that is larger than a.
    intervals = []
    for ii,A_idx_lst in enumerate(A):
        B_idx_lst = [bisect.bisect_left(B[ii], idx) for idx in A_idx_lst]
        chan_intervals = []
        for jj,idx_l in enumerate(B_idx_lst):
            if idx_l == len(B[ii]):
                chan_intervals.append((A_idx_lst[jj],endIdx))
            else:
                chan_intervals.append((A_idx_lst[jj],B[ii][idx_l]))
        intervals.append(chan_intervals)
        # previous code already takes care of the [] possibility
        #if B_idx_lst == []:
        #    intervals.append([])
    return intervals



################################################
#	Complexity features
################################################    

##########
# Extract the Shannon Entropy
# threshold the signal and make it discrete, normalize it and then compute entropy
def shannonEntropy(eegData, bin_min, bin_max, binWidth, fs, lowcut, highcut):
    H = np.zeros((eegData.shape[0], eegData.shape[2]))
    for chan in range(H.shape[0]):
        for epoch in range(H.shape[1]):
            counts, binCenters = np.histogram(eegData[chan,:,epoch], bins=np.arange(bin_min+1, bin_max, binWidth))
            nz = counts > 0
            prob = counts[nz] / np.sum(counts[nz])
            H[chan, epoch] = -np.dot(prob, np.log2(prob/binWidth))
    return H


def ShannonEntropy(eegData, bin_min, bin_max, binWidth, fs):
    H = np.zeros((eegData.shape[0], eegData.shape[2]))
    for chan in range(H.shape[0]):
        for epoch in range(H.shape[1]):
            counts, binCenters = np.histogram(eegData[chan,:,epoch], bins=np.arange(bin_min+1, bin_max, binWidth))
            nz = counts > 0
            prob = counts[nz] / np.sum(counts[nz])
            H[chan, epoch] = -np.dot(prob, np.log2(prob/binWidth))
    return H
    
##########
# Extract the tsalis Entropy
# def tsalisEntropy(eegData, bin_min, bin_max, binWidth, orders = [1]):
#     H = [np.zeros((eegData.shape[0], eegData.shape[2]))]*len(orders)
#     for chan in range(H[0].shape[0]):
#         for epoch in range(H[0].shape[1]):
#             counts, bins = np.histogram(eegData[chan,:,epoch], bins=np.arange(-200+1, 200, 2))
#             dist = dit.Distribution([str(bc).zfill(5) for bc in bins[:-1]],counts/sum(counts))
#             for ii,order in enumerate(orders):
#                 H[ii][chan,epoch] = tsallis_entropy(dist,order)
    # return H


# ##########
# # Cepstrum Coefficients (n=2)
# def mfcc(eegData, fs, order=2):
#     n_channels, n_times, n_epochs = eegData.shape
#     H = np.zeros((n_channels, n_epochs, order))
    
#     H = np.zeros((eegData.shape[0], eegData.shape[2], order))
#     for chan in range(H.shape[0]):
#         for epoch in range(H.shape[1]):
#             mfcc_features = librosa.feature.mfcc(
#                 np.asfortranarray(eegData[chan, :, epoch]), sr=fs
#             )[:order].T  # Extract the first 'order' coefficients
#             H[chan, epoch, :] = mfcc_features

#     return H


##########
# Lyapunov exponent
def lyapunov(eegData):
    return np.mean(np.log(np.abs(np.gradient(eegData,axis=1))),axis=1)
    
##########
# Fractal Embedding Dimension
# From pyrem: packadge for sleep scoring from EEG data
# https://github.com/gilestrolab/pyrem/blob/master/src/pyrem/univariate.py
# def hFD(a, k_max): #Higuchi FD
#     L = []
#     x = []
#     N = len(a)

#     for k in range(1,k_max):
#         Lk = 0
#         for m in range(0,k):
#             #we pregenerate all idxs
#             idxs = np.arange(1,int(np.floor((N-m)/k)),dtype=np.int32)
#             Lmk = np.sum(np.abs(a[m+idxs*k] - a[m+k*(idxs-1)]))
#             Lmk = (Lmk*(N - 1)/(((N - m)/ k)* k)) / k
#             Lk += Lmk

#         L.append(np.log(Lk/(m+1)))
#         x.append([np.log(1.0/ k), 1])

#     (p, r1, r2, s)=np.linalg.lstsq(x, L, rcond=None)
#     return p[0]


def hFD(eegData, k_max):
    n_channels, n_samples, n_epochs = eegData.shape
    L = []
    x = []

    for k in range(1, k_max):
        Lk = 0
        for m in range(0, k):
            idxs = np.arange(1, int(np.floor((n_samples - m) / k)), dtype=np.int32)
            Lmk = 0
            for epoch in range(n_epochs):
                Lmk += np.sum(np.abs(eegData[:, m + idxs * k, epoch] - eegData[:, m + k * (idxs - 1), epoch]))
            Lmk = (Lmk * (n_samples - 1) / (((n_samples - m) / k) * k)) / (k * n_epochs)
            Lk += Lmk

        L.append(np.log(Lk / (m + 1)))
        x.append([np.log(1.0 / k), 1])

    x = np.array(x)
    L = np.array(L)

    (p, _, _, _) = np.linalg.lstsq(x, L, rcond=None)
    return p[0]
    
##########
# Hjorth Mobility
# Hjorth Complexity
# variance = mean(signal^2) iff mean(signal)=0
# which it is be because I normalized the signal
# Assuming signals have mean 0
# Mobility = sqrt( mean(dx^2) / mean(x^2) )
def hjorthParameters(xV):
    dxV = np.diff(xV, axis=1)
    ddxV = np.diff(dxV, axis=1)

    mx2 = np.mean(np.square(xV), axis=1)
    mdx2 = np.mean(np.square(dxV), axis=1)
    mddx2 = np.mean(np.square(ddxV), axis=1)

    mob = mdx2 / mx2
    complexity = np.sqrt((mddx2 / mdx2) / mob)
    mobility = np.sqrt(mob)

    # PLEASE NOTE that Mohammad did NOT ACTUALLY use hjorth complexity,
    # in the matlab code for hjorth complexity subtraction by mob not division was used 
    return mobility, complexity

##########
# false nearest neighbor descriptor
def falseNearestNeighbor(eegData, fast=True):
    # Average Mutual Information
    # There exist good arguments that if the time delayed mutual
    # information exhibits a marked minimum at a certain value of tex2html_wrap_inline6553,
    # then this is a good candidate for a reasonable time delay.
    npts = 1000   # not sure about this?
    maxdims = 50
    max_delay = 2 # max_delay = 200  # TODO: need to use 200, but also need to speed this up
    distance_thresh = 0.5
    
    out = np.zeros((eegData.shape[0], eegData.shape[2]))
    for chan in range(eegData.shape[0]):
        for epoch in range(eegData.shape[2]):
            if fast:
                out[chan, epoch] = 0
            else:
                cur_eegData = eegData[chan, :, epoch]
                lagidx = 0  # we are looking for the index of the lag that makes the signal maximally uncorrelated to the original
                # # minNMI = 1  # normed_mutual_info is from 1 (perfectly correlated) to 0 (not at all correlated) 
                # # for lag in range(1, max_delay):
                # #     x = cur_eegData[:-lag]
                # #     xlag = cur_eegData[lag:]
                # #     # convert float data into histogram bins
                # #     nbins = int(np.floor(1 + np.log2(len(x)) + 0.5))
                # #     x_discrete = np.histogram(x, bins=nbins)[0]
                # #     xlag_discrete = np.histogram(xlag, bins=nbins)[0]
                # #     cNMI = normed_mutual_info(x_discrete, xlag_discrete)
                # #     if cNMI < minNMI:
                # #         minNMI = cNMI
                # #         lagidx = lag
                # nearest neighbors part
                knn = int(max(2, 6*lagidx))  # heuristic (number of nearest neighbors to look up)
                m = 1 # lagidx + 1

                # y is the embedded version of the signal
                y = np.zeros((maxdims+1, npts))
                for d in range(maxdims+1):
                    tmp = cur_eegData[d*m:d*m + npts]
                    y[d, :tmp.shape[0]] = tmp
                
                nnd = np.ones((npts, maxdims))
                nnz = np.zeros((npts, maxdims))
                
                # see where it tends to settle
                for d in range(1, maxdims):
                    for k in range(0, npts):
                        # get the distances to all points in the window (distance given embedding dimension)
                        dists = []
                        for nextpt in range(1, knn+1):
                            if k+nextpt < npts:
                                dists.append(np.linalg.norm(y[:d, k] - y[:d, k+nextpt]))
                        if len(dists) > 0:
                            minIdx = np.argmin(dists)
                            if dists[minIdx] == 0:
                                dists[minIdx] = 0.0000001  # essentially 0 just silence the error
                            nnd[k, d-1] = dists[minIdx]
                            nnz[k, d-1] = np.abs( y[d+1, k] - y[d+1, minIdx+1+k] )
                # aggregate results
                mindim = np.mean(nnz/nnd > distance_thresh, axis=0) < 0.1
                # get the index of the first occurence of the value true
                # (a 1 in the binary representation of true and false)
                out[chan, epoch] = np.argmax(mindim)
        
    return out 

##########
# ARMA coefficients
def arma(eegData,orders=(2,1)):
    """
    Fit ARMA models to EEG data for multiple channels and epochs.

    Parameters:
    - eegData (numpy.ndarray): EEG data with shape (n_channels, n_times, n_epochs).
    - orders (list of tuples): List of (p, q) orders for ARMA modeling for each channel and epoch.

    Returns:
    - ar_params (numpy.ndarray): AR parameters for each channel and epoch (shape: n_channels, n_epochs, p).
    """
    n_channels, n_times, n_epochs = eegData.shape
    p, q = orders
    H = np.zeros((n_channels, n_epochs, p + q))

    for chan in range(n_channels):
        for epoch in range(n_epochs):
            arma_mod = sm.tsa.ARIMA(eegData[chan, :, epoch], order=(p, 0, q))
            arma_res = arma_mod.fit(trend='nc', disp=-1)
            H[chan, epoch, :p] = arma_res.arparams
            H[chan, epoch, p:] = arma_res.maparams

    return H
