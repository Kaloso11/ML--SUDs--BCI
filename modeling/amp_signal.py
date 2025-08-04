#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 21:32:50 2023

@author: kaloso
"""

import mne
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, filtfilt
from scipy.signal import welch

#%%

# # Load the epochs
epochs = mne.read_epochs('epoch_data14-epo.fif', preload=True)

# Load ICA data from the .npy file
# ica_data = np.load('ica_data6.npy')

# Extract the first epoch
# first_epoch_data = ica_data[:, :, -1] 

# Get data for the first and last epoch
data = np.squeeze(epochs[-1].get_data())  # First epoch
# data2 = np.squeeze(epochs[-1].get_data())  # Last epoch
        
        
#%%


# Generate a time vector based on your sampling rate and the length of your epochs
sfreq = 128  # Example: 100 Hz

# Define the band frequency range
freq_band = (25, 55)

# Design a bandpass filter for the Delta band
b, a = butter(N=4, Wn=np.array(freq_band) / (sfreq / 2), btype='band')


# Apply the filter to the data
filtered_data = filtfilt(b, a, data, axis=1)


# Generate a time vector based on the sampling rate and the length of your epochs
time = np.linspace(0, data.shape[1] / sfreq, data.shape[1])

#%%


# global_min, global_max = np.inf, -np.inf

# # Assuming you have a list of frequency bands to analyze
# freq_bands = [(0.5, 4), (4, 8), (8, 12), (12, 25), (25, 55)]  # Example bands

# for freq_band in freq_bands:
#     # Apply the bandpass filter for each band (repeat the filtering process)
#     b, a = butter(N=4, Wn=np.array(freq_band) / (sfreq / 2), btype='band')
#     filtered_data = filtfilt(b, a, data, axis=1)
#     average_signal = np.mean(filtered_data, axis=0)

#     # Update global min/max
#     band_min, band_max = np.min(average_signal), np.max(average_signal)
#     global_min = min(global_min, band_min)
#     global_max = max(global_max, band_max)


#%%

average_signal = np.mean(filtered_data, axis=0)


# Plotting the average EEG signal amplitude
plt.figure(figsize=(12, 6))
plt.plot(time, average_signal)
plt.ylim(-200,200)  # Consistent y-axis range
plt.title('Average EEG Signal Amplitude Over Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude (µV)')
plt.grid(True)
plt.show()


#%%

# # n_samples = first_epoch_data.shape[1]
# # time = np.linspace(0, n_samples / sfreq, n_samples)

# # Calculate average signal amplitude across all components
# average_amplitude = np.mean(filtered_data, axis=0)

# # Plotting average signal amplitude
# plt.figure(figsize=(10, 4))
# plt.plot(time, average_amplitude)
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.title('Average Signal Amplitude Across All ICA Components')
# plt.grid(True)
# plt.show()


#%%


# # Compute PSD for each component and average
# psds = []
# for component in first_epoch_data:
#     frequencies, psd = welch(component, fs=sfreq)
#     psds.append(psd)

# average_psd = np.mean(psds, axis=0)

# # Plotting average PSD
# plt.figure(figsize=(10, 4))
# plt.semilogy(frequencies, average_psd)
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Power Spectral Density (dB/Hz)')
# plt.title('Average Power Spectral Density Across All ICA Components - First Epoch')
# plt.grid(True)
# plt.show()