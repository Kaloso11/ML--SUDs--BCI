#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 14:01:32 2023

@author: kaloso
"""

import EEGExtract_Continue as eegcont
import numpy as np
# import numba
# from numba import cuda
#from feature_extraction_complex import eegData_delta


import warnings
#mne.set_log_level("WARNING")
warnings.filterwarnings("ignore")


# %% Load data

eegData = np.load('batched_data1.npy')

fs = 128

# %% Feature extraction


###############################################
#	Continuity features
################################################ 

# Median Frequency
medianFreqRes = eegcont.medianFreq(eegData,fs)

#band_power = eegcont.bandPower(eegData, fs, lowcut, highcut)

# δ band Power
bandPwr_delta = eegcont.deltaBandPower(eegData, fs)
# θ band Power
bandPwr_theta = eegcont.thetaBandPower(eegData, fs)
# α band Power
bandPwr_alpha = eegcont.alphaBandPower(eegData, fs)
# β band Power
bandPwr_beta = eegcont.betaBandPower(eegData, fs)
# γ band Power
bandPwr_gamma = eegcont.gammaBandPower(eegData, fs)

std_res = eegcont.eegStd(eegData)

# α/δ Ratio
#ratio_res = eegcont.eegRatio(eegData,fs)

# ratio_delta_theta = eegcont.deltaThetaRatio(eegData,fs)

# ratio_theta_alpha = eegcont.thetaAlphaRatio(eegData,fs)

# ratio_beta_alpha = eegcont.alphaBetaRatio(eegData,fs)

# Regularity (burst-suppression)
regularity_res = eegcont.eegRegularity(eegData,fs)


# Voltage < 5μ
volt05_res = eegcont.eegVoltage(eegData,voltage=5)
# Voltage < 10μ
volt10_res = eegcont.eegVoltage(eegData,voltage=10)
# Voltage < 20μ
volt20_res = eegcont.eegVoltage(eegData,voltage=20)

# # Diffuse Slowing
# df_res = eegcont.diffuseSlowing(eegData)

# Spikes
minNumSamples = int(70*fs/1000)
spikeNum_res = eegcont.spikeNum(eegData,minNumSamples)

# Delta burst after Spike
#deltaBurst_res = eegcont.burstAfterSpike(eegData,eegData_delta,minNumSamples=7,stdAway = 3)
# Sharp spike
sharpSpike_res = eegcont.shortSpikeNum(eegData,minNumSamples)
# Number of Bursts
numBursts_res = eegcont.numBursts(eegData,fs)
# # Burst length μ and σ
# burstLenMean_res,burstLenStd_res = eegcont.burstLengthStats(eegData,fs,bursts)
# Burst Band Power for δ band
burstBandPwrDelta = eegcont.burstBandPowers(eegData, 0.5, 4, fs)
# Burst Band Power for α band
burstBandPwrTheta = eegcont.burstBandPowers(eegData, 4, 8, fs)
# Burst Band Power for θ band
burstBandPwrAlpha = eegcont.burstBandPowers(eegData, 8, 12, fs)
# Burst Band Power for β band
burstBandPwrBeta = eegcont.burstBandPowers(eegData, 12, 25, fs)
# Burst Band Power for γ band
burstBandPwrGamma = eegcont.burstBandPowers (eegData, 25, 55, fs)
# Number of Suppressions
numSupps_res = eegcont.numSuppressions(eegData,fs)
# Suppression length μ and σ
#suppLenMean_res,suppLenStd_res = eegcont.suppressionLengthStats(eegData,fs)


# %% Store the results

feature_list = []
feature_list.append(medianFreqRes)
feature_list.append(bandPwr_delta)
feature_list.append(bandPwr_theta)
feature_list.append(bandPwr_alpha)
feature_list.append(bandPwr_beta)
feature_list.append(bandPwr_gamma)
feature_list.append(std_res)
feature_list.append(regularity_res)
feature_list.append(volt05_res)
feature_list.append(volt10_res)
feature_list.append(volt20_res)
feature_list.append(spikeNum_res)
feature_list.append(sharpSpike_res)
feature_list.append(numBursts_res)
feature_list.append(burstBandPwrDelta)
feature_list.append(burstBandPwrTheta)
feature_list.append(burstBandPwrAlpha)
feature_list.append(burstBandPwrBeta)
feature_list.append(burstBandPwrGamma)
feature_list.append(numSupps_res)
#feature_list.append(FalseNnRes)


continue_features = np.vstack(feature_list).transpose()



if __name__ == "__main__":
    
    np.save('continue_features1.npy', continue_features)


# %%
