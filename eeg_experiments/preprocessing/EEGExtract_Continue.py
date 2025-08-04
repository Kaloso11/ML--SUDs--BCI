#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 15:46:18 2023

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
#import numba
#from numba import jit, njit, cuda
import torch
import librosa



################################################
#	Auxiliary Functions
################################################

##########
# Filter the eegData, midpass filter 
#	eegData: 3D nrray [chans x ms x epochs] 
# def filt_data(eegData, lowcut, highcut, fs, order=7):
#     nyq = 0.5 * fs
#     low = lowcut / nyq
#     high = highcut / nyq
#     b, a = signal.butter(order, [low, high], btype='band')
#     filt_eegData = signal.lfilter(b, a, eegData, axis = 1)
#     return filt_eegData

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


# # Detect bursts and supressions in eeg data
# def burst_supression_detection(eegData,fs,suppression_threshold = 10):
#  	'''
#  	# DETECT EMG ARTIFACTS.
#  	nyq = 0.5 * fs
#  	low = low / nyq
#  	high = high / nyq
#  	be, ae = signal.butter(order, [low, high], btype='band')
#  	'''
#  	# CALCULATE ENVELOPE
#  	e = abs(signal.hilbert(eegData,axis=1));
#  	# same as smooth(e,Fs/4) in MATLAB, apply 1/2 second smoothing
#  	ME = np.array([np.convolve(el,np.ones(int(fs/4))/(fs/4),'same') for el in e.tolist()])
#  	e = ME
#  	# DETECT SUPRESSIONS
#  	# apply threshold -- 10uv
#  	z = (ME<suppression_threshold)
#  	# remove too-short suppression segments
#  	z = fcnRemoveShortEvents(z,fs/2)
#  	# remove too-short burst segments
#  	b = fcnRemoveShortEvents(1-z,fs/2)
#  	z = 1-b
#  	went_high = [np.where(np.array(chD[:-1]) < np.array(chD[1:]))[0].tolist() for chD in z.tolist()]
#  	went_low = [np.where(np.array(chD[:-1]) > np.array(chD[1:]))[0].tolist() for chD in z.tolist()]

#  	bursts = get_intervals(went_high,went_low)
#  	supressions = get_intervals(went_low,went_high)

#  	return bursts,supressions
 
    
 
def calculate_envelope(eegData, fs, smoothing_window=5):
    envelope = abs(signal.hilbert(eegData, axis=1))
    
    if envelope.size == 0:
        # Handle the case where the envelope is empty
        return np.zeros_like(eegData)
    
    smoothed_envelope = np.array([np.convolve(el, np.ones(smoothing_window) / smoothing_window, 'same') for el in envelope.tolist()])
    
    return smoothed_envelope

def detect_suppressions(envelope, suppression_threshold):
    # Detect suppressions based on the threshold
    suppressions = envelope < suppression_threshold
    
    return suppressions

def burst_suppression_detection(eegData, fs, suppression_threshold=10):
    # Calculate the envelope
    envelope = calculate_envelope(eegData, fs)
    
    # Detect suppressions
    suppressions = detect_suppressions(envelope, suppression_threshold)
    
    # Remove too-short suppression segments
    suppressions = fcnRemoveShortEvents(suppressions, fs / 2)
    
    # Remove too-short burst segments
    bursts = fcnRemoveShortEvents(1 - suppressions, fs / 2)
    
    return bursts, suppressions    



################################################
#	Continuity features
################################################  

##########
# median frequency
def medianFreq(eegData,fs):
    H = np.zeros((eegData.shape[0], eegData.shape[2]))
    for chan in range(H.shape[0]):
        freqs, powers = signal.periodogram(eegData[chan, :, :], fs, axis=0)
        H[chan,:] = freqs[np.argsort(powers,axis=0)[len(powers)//2]]
    return H

##########
# calculate band power
# def bandPower(eegData, lowcut, highcut, fs):
# 	eegData_band = filt_data(eegData, lowcut, highcut, fs, order=7)
# 	freqs, powers = signal.periodogram(eegData_band, fs, axis=1)
# 	bandPwr = np.mean(powers,axis=1)
# 	return bandPwr

def bandPower(eegData, fs, lowcut, highcut):
    num_channels, num_samples, num_epochs = eegData.shape
    bandPwr = np.zeros((num_channels, num_epochs))
    
    for epoch in range(num_epochs):
        for chan in range(num_channels):
            # Extract the EEG data for the current channel and epoch
            epoch_data = eegData[chan, :, epoch]
            
            # Calculate the power spectrum for the current epoch and channel
            freqs, powers = signal.periodogram(epoch_data, fs)
            
            # Calculate the band power (average power) for this epoch and channel
            bandPwr[chan, epoch] = np.mean(powers)
    
    return bandPwr


def deltaBandPower(eegData, fs):
    return bandPower(eegData, fs, 0.5, 4)

def thetaBandPower(eegData, fs):
    return bandPower(eegData, fs, 4, 8)

def alphaBandPower(eegData, fs):
    return bandPower(eegData, fs, 8, 12)

def betaBandPower(eegData, fs):
    return bandPower(eegData, fs, 12, 25)

def gammaBandPower(eegData, fs):
    return bandPower(eegData, fs, 25, 55)


##########
# numberOfSpikes    
def spikeNum(eegData,minNumSamples=7,stdAway = 3):
    H = np.zeros((eegData.shape[0], eegData.shape[2]))
    for chan in range(H.shape[0]):
        for epoch in range(H.shape[1]):
            mean = np.mean(eegData[chan, :, epoch])
            std = np.std(eegData[chan,:,epoch],axis=1)
            H[chan,epoch] = len(signal.find_peaks(abs(eegData[chan,:,epoch]-mean), 3*std,epoch,width=7)[0])
    return H

##########    
# Standard Deviation
def eegStd(eegData):
	std_res = np.std(eegData,axis=1)
	return std_res

##########
# α/δ Ratio
# def eegRatio(eegData,fs):
#     # delta (0.5–4 Hz)
# 	eegData_delta = filt_data(eegData, 0.5, 4, fs)
# 	# alpha (8–12 Hz)
# 	eegData_alpha = filt_data(eegData, 8, 12, fs)
# 	# calculate the power
# 	powers_alpha = bandPower(eegData, 8, 12, fs)
# 	powers_delta = bandPower(eegData, 0.5, 4, fs)
# 	ratio_res = np.sum(powers_alpha,axis=0) / np.sum(powers_delta,axis=0)
# 	return np.expand_dims(ratio_res, axis=0)



def deltaThetaRatio(eegData, fs):
    eegData_delta = eegData, 0.5, 4, fs
    eegData_theta = eegData, 4, 8, fs
    
    powers_delta = bandPower(eegData_delta, 0.5, 4, fs)
    powers_theta = bandPower(eegData_theta, 4, 8, fs)
    
    ratio_res = np.sum(powers_theta, axis=0) / np.sum(powers_delta, axis=0)
    
    return np.expand_dims(ratio_res, axis=0)

def thetaAlphaRatio(eegData, fs):
    eegData_theta = eegData, 4, 8, fs
    eegData_alpha = eegData, 8, 12, fs
    
    powers_theta = eegData_theta, 4, 8, fs
    powers_alpha = eegData_alpha, 8, 12, fs
    
    ratio_res = np.sum(powers_alpha, axis=0) / np.sum(powers_theta, axis=0)
    
    return np.expand_dims(ratio_res, axis=0)

def alphaBetaRatio(eegData, fs):
    eegData_alpha = eegData, 8, 12, fs
    eegData_beta =  eegData, 12, 25, fs
    
    powers_alpha = bandPower(eegData_alpha, 8, 12, fs)
    powers_beta = bandPower(eegData_beta, 12, 25, fs)
    
    ratio_res = np.sum(powers_alpha, axis=0) / np.sum(powers_beta, axis=0)
    
    return np.expand_dims(ratio_res, axis=0)

###########
# Regularity (burst-suppression)
# Regularity of eeg
# filter with a window of 0.5 seconds to create a nonnegative smooth signal.
# In this technique, we first squared the signal and applied a moving-average
# The window length of the moving average was set at 0.5 seconds.
def eegRegularity(eegData, Fs=128):
    in_x = np.square(eegData)  # square signal
    num_wts = Fs//2  # find the filter length in samples - we want 0.5 seconds.
    q = signal.lfilter(np.ones(num_wts) / num_wts, 1, in_x, axis=1)
    q = -np.sort(-q, axis=1) # descending sort on smooth signal
    N = q.shape[1]
    u2 = np.square(np.arange(1, N+1))
    # COMPUTE THE Regularity
    # dot each 5min epoch with the quadratic data points and then normalize by the size of the dotted things    
    reg = np.sqrt( np.einsum('ijk,j->ik', q, u2) / (np.sum(q, axis=1)*(N**2)/3) )
    return reg

###########
# Voltage < (5μ, 10μ, 20μ)
def eegVoltage(eegData,voltage=20):
	eegFilt = eegData.copy()
	eegFilt[abs(eegFilt) > voltage] = np.nan
	volt_res = np.nanmean(eegFilt,axis=1)
	return volt_res

##########
# # Diffuse Slowing
# # look for diffuse slowing (bandpower max from frequency domain integral)
# # repeated integration of a huge tensor is really expensive
# def diffuseSlowing(eegData, Fs=100, fast=True):
#     maxBP = np.zeros((eegData.shape[0], eegData.shape[2]))
#     idx = np.zeros((eegData.shape[0], eegData.shape[2]))
#     if fast:
#         return idx
#     for j in range(1, Fs//2):
#         print("BP", j)
#         cbp = bandpower(eegData, Fs, [j-1, j])
#         biggerCIdx = cbp > maxBP
#         idx[biggerCIdx] = j
#         maxBP[biggerCIdx] = cbp[biggerCIdx]
#     return (idx < 8)

##########
# Spikes
def spikeNum(eegData,minNumSamples=7,stdAway = 3):
    H = np.zeros((eegData.shape[0], eegData.shape[2]))
    for chan in range(H.shape[0]):
        for epoch in range(H.shape[1]):
            mean = np.mean(eegData[chan, :, epoch])
            std = np.std(eegData[chan,:,epoch])
            H[chan,epoch] = len(signal.find_peaks(abs(eegData[chan,:,epoch]-mean), 3*std,epoch,width=7)[0])
    return H

##########
# Delta Burst after spike
def burstAfterSpike(eegData,eegData_subband,minNumSamples=7,stdAway = 3):
    H = np.zeros((eegData.shape[0], eegData.shape[2]))
    for chan in range(H.shape[0]):
        for epoch in range(H.shape[1]):
            preBurst = 0
            postBurst = 0
            mean = np.mean(eegData[chan, :, epoch])
            std = np.std(eegData[chan,:,epoch])
            idxList = signal.find_peaks(abs(eegData[chan,:,epoch]-mean), stdAway*std,epoch,width=minNumSamples)[0]
            for idx in idxList:
                preBurst += np.mean(eegData_subband[chan,idx-7:idx-1,epoch])
                postBurst += np.mean(eegData_subband[chan,idx+1:idx+7,epoch])
            H[chan,epoch] = postBurst - preBurst
    return H

##########
# Sharp spike
def shortSpikeNum(eegData,minNumSamples=7,stdAway = 3):
    H = np.zeros((eegData.shape[0], eegData.shape[2]))
    for chan in range(H.shape[0]):
        for epoch in range(H.shape[1]):
            mean = np.mean(eegData[chan, :, epoch])
            std = np.std(eegData[chan,:,epoch])
            longSpikes = set(signal.find_peaks(abs(eegData[chan,:,epoch]-mean), 3*std,epoch,width=7)[0])
            shortSpikes = set(signal.find_peaks(abs(eegData[chan,:,epoch]-mean), 3*std,epoch,width=1)[0])
            H[chan,epoch] = len(shortSpikes.difference(longSpikes))
    return H

##########
# Number of Bursts
def numBursts(eegData,fs):
	bursts = []
	supressions = []
	for epoch in range(eegData.shape[2]):
		epochBurst,epochSupressions = burst_suppression_detection(eegData[:,:,epoch],fs,suppression_threshold=10)#,low=30,high=49)
		bursts.append(epochBurst)
		supressions.append(epochSupressions)
	# Number of Bursts
	numBursts_res = np.zeros((eegData.shape[0], eegData.shape[2]))
	for chan in range(numBursts_res.shape[0]):
		for epoch in range(numBursts_res.shape[1]):
			numBursts_res[chan,epoch] = len(bursts[epoch][chan])
	return numBursts_res
	
##########
# # Burst length μ and σ
# def burstLengthStats(eegData,fs):
# 	bursts = []
# 	supressions = []
# 	for epoch in range(eegData.shape[2]):
# 		epochBurst,epochSupressions = burst_suppression_detection(eegData[:,:,epoch],fs,suppression_threshold=10)#,low=30,high=49)
# 		bursts.append(epochBurst)
# 		supressions.append(epochSupressions)
# 	# Number of Bursts
# 	burstMean_res = np.zeros((eegData.shape[0], eegData.shape[2]))
# 	burstStd_res = np.zeros((eegData.shape[0], eegData.shape[2]))
# 	for chan in range(burstMean_res.shape[0]):
# 		for epoch in range(burstMean_res.shape[1]):
# 			burstMean_res[chan,epoch] = np.mean([burst[1]-burst[0] for burst in bursts[epoch][chan]])
# 			burstStd_res[chan,epoch] = np.std([burst[1]-burst[0] for burst in bursts[epoch][chan]])
# 	burstMean_res = np.nan_to_num(burstMean_res)
# 	burstStd_res = np.nan_to_num(burstStd_res)
# 	return burstMean_res,burstStd_res


def burstLengthStats(eegData, fs, bursts):
    bursts_mean = np.zeros((eegData.shape[0], eegData.shape[2]))
    bursts_std = np.zeros((eegData.shape[0], eegData.shape[2]))

    for chan in range(eegData.shape[0]):
        for epoch in range(eegData.shape[2]):
            # Extract burst intervals for the current channel and epoch
            burst_intervals = bursts[chan][epoch]

            # Calculate the mean and standard deviation of burst lengths
            if len(burst_intervals) > 0:
                burst_lengths = [burst[1] - burst[0] for burst in burst_intervals]
                bursts_mean[chan, epoch] = np.mean(burst_lengths)
                bursts_std[chan, epoch] = np.std(burst_lengths)
            else:
                # Handle the case where there are no bursts for this channel and epoch
                bursts_mean[chan, epoch] = 0.0
                bursts_std[chan, epoch] = 0.0

    return bursts_mean, bursts_std


##########
# Burst band powers (δ, α, θ, β, γ)
# def burstBandPowers(eegData, lowcut, highcut, fs, order=7):
# 	band_burst_powers = np.zeros((eegData.shape[0], eegData.shape[2]))
# 	bursts = []
# 	supressions = []
# 	for epoch in range(eegData.shape[2]):
# 		epochBurst,epochSupressions = burst_suppression_detection(eegData[:,:,epoch],fs,suppression_threshold=10)#,low=30,high=49)
# 		bursts.append(epochBurst)
# 		supressions.append(epochSupressions)
# 	eegData_band = filt_data(eegData, lowcut, highcut, fs, order=7)
# 	for epoch,epochBursts in enumerate(bursts):
# 		for chan,chanBursts in enumerate(epochBursts):
# 			epochPowers = []  
# 			for burst in chanBursts:
# 				if burst[1] == eegData.shape[1]:
# 					burstData =  eegData_band[:,burst[0]:,epoch]
# 				else:
# 					burstData =  eegData_band[:,burst[0]:burst[1],epoch]
# 				freqs, powers = signal.periodogram(burstData, fs, axis=1)
# 				epochPowers.append(np.mean(powers,axis=1))
# 			band_burst_powers[chan,epoch] = np.mean(epochPowers)	
# 	return band_burst_powers


# def burstBandPowers(eegData, fs,lowcut,highcut, suppression_threshold=10):
#     band_burst_powers = np.zeros((eegData.shape[0], eegData.shape[2]))
#     bursts = []
    
#     for epoch in range(eegData.shape[2]):
#         epochBurst, epochSupressions = burst_suppression_detection(eegData[:,:,epoch], fs, suppression_threshold)
#         bursts.append(epochBurst)
    
#     for epoch, epochBursts in enumerate(bursts):
#         for chan, chanBursts in enumerate(epochBursts):
#             epochPowers = []
            
#             for burst in chanBursts:
#                 if burst[1] == eegData.shape[1]:
#                     burstData = eegData[:, burst[0]:, epoch]
#                 else:
#                     burstData = eegData[:, burst[0]:burst[1], epoch]
                
#                 # Calculate the power spectrum for the burst data
#                 freqs, powers = signal.periodogram(burstData, fs, axis=1)
#                 epochPowers.append(np.mean(powers, axis=1))
            
#             band_burst_powers[chan, epoch] = np.mean(epochPowers)
    
#     return band_burst_powers


def burstBandPowers(eegData, fs,lowcut,highcut, suppression_threshold=10):
    band_burst_powers = np.zeros((eegData.shape[0], eegData.shape[2]))
    bursts = []

    for epoch in range(eegData.shape[2]):
        epochBurst, epochSupressions = burst_suppression_detection(eegData[:, :, epoch], fs, suppression_threshold)
        bursts.append(epochBurst)

    for epoch, epochBursts in enumerate(bursts):
        for chan, chanBursts in enumerate(epochBursts):
            epochPowers = []

            for burst in chanBursts:
                if isinstance(burst, (list, np.ndarray)) and len(burst) == 2:
                    if burst[1] == eegData.shape[1]:
                        burstData = eegData[:, burst[0]:, epoch]
                    else:
                        burstData = eegData[:, burst[0]:burst[1], epoch]

                    freqs, powers = signal.periodogram(burstData, fs, axis=1)
                    epochPowers.append(np.mean(powers, axis=1))

            band_burst_powers[chan, epoch] = np.mean(epochPowers)

    return band_burst_powers



##########
# Number of Suppressions
def numSuppressions(eegData,fs,suppression_threshold=10):
	bursts = []
	supressions = []
	for epoch in range(eegData.shape[2]):
		epochBurst,epochSupressions = burst_suppression_detection(eegData[:,:,epoch],fs,suppression_threshold=suppression_threshold)#,low=30,high=49)
		bursts.append(epochBurst)
		supressions.append(epochSupressions)
	numSupprs_res = np.zeros((eegData.shape[0], eegData.shape[2]))
	for chan in range(numSupprs_res.shape[0]):
		for epoch in range(numSupprs_res.shape[1]):
			numSupprs_res[chan,epoch] = len(supressions[epoch][chan])
	return numSupprs_res

##########
# Suppression length μ and σ
def suppressionLengthStats(eegData,fs,suppression_threshold=10):
	bursts = []
	supressions = []
	for epoch in range(eegData.shape[2]):
		epochBurst,epochSupressions = burst_suppression_detection(eegData[:,:,epoch],fs,suppression_threshold=suppression_threshold)#,low=30,high=49)
		bursts.append(epochBurst)
		supressions.append(epochSupressions)
	supressionMean_res = np.zeros((eegData.shape[0], eegData.shape[2]))
	supressionStd_res = np.zeros((eegData.shape[0], eegData.shape[2]))
	for chan in range(supressionMean_res.shape[0]):
		for epoch in range(supressionMean_res.shape[1]):
			supressionMean_res[chan,epoch] = np.mean([suppr[1]-suppr[0] for suppr in supressions[epoch][chan]])
			supressionStd_res[chan,epoch] = np.std([suppr[1]-suppr[0] for suppr in supressions[epoch][chan]])
	supressionMean_res = np.nan_to_num(supressionMean_res)
	supressionStd_res = np.nan_to_num(supressionStd_res)
	return supressionMean_res, supressionStd_res
