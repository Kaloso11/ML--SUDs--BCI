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

#%%

fif_files = [f'epoch_data{i}-epo.fif' for i in range(1, 18)]

# Initialize lists to store epochs for each condition
epochs_before_alcohol = []
epochs_after_alcohol = []

# Load and extract the specific epochs from each file, excluding subjects with less than 6 epochs
for file in fif_files:
    epochs = mne.read_epochs(file, preload=True)
    
    # Check if the subject has at least 6 epochs
    if len(epochs) >= 6:
        # Extracts the first epoch (before alcohol) and squeezes it to 2D
        epoch_before = np.squeeze(epochs[0].get_data())
        # Extracts the sixth epoch (end of alcohol consumption) and squeezes it to 2D
        epoch_after = np.squeeze(epochs[5].get_data())
        
        epochs_before_alcohol.append(epoch_before)
        epochs_after_alcohol.append(epoch_after)
    else:
        print(f"Skipping file {file} as it has less than 6 epochs.")
        
#%%


# Function to apply a bandpass filter
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

# Define Delta band frequency range
delta_band = (0.5, 4)
        
        
#%%        

# You can now average across these 2D arrays
erp_before_alcohol = np.mean(epochs_before_alcohol, axis=0)
erp_after_alcohol = np.mean(epochs_after_alcohol, axis=0)

# Now erp_before_alcohol and erp_after_alcohol contain the averaged ERP waveforms for each condition

# Generate a time vector based on your sampling rate and the length of your epochs
# Replace 'sampling_rate' and 'n_timepoints' with your actual values
sampling_rate = 128  # Example: 100 Hz
n_timepoints = erp_before_alcohol.shape[1]  # Number of time points in your epoch
time = np.linspace(0, n_timepoints / sampling_rate, n_timepoints)

# # Plot ERP before alcohol consumption
# plt.figure(figsize=(10, 4))
# plt.plot(time, erp_before_alcohol.T)
# plt.title('ERP Before Alcohol Consumption')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude (µV)')
# plt.show()

# # Plot ERP after alcohol consumption
# plt.figure(figsize=(10, 4))
# plt.plot(time, erp_after_alcohol.T)
# plt.title('ERP After Alcohol Consumption')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude (µV)')
# plt.show()

#%% connectivity heatmap

# data = np.array(erp_after_alcohol)  # Assuming this is a 2D array (channels, samples)

# # Calculate the connectivity matrix using correlation
# connectivity_matrix = np.corrcoef(data)

# # Plotting the connectivity heatmap
# plt.figure(figsize=(10, 8))
# sns.heatmap(connectivity_matrix, cmap='viridis', square=True)
# plt.title('Connectivity Heatmap')
# plt.xlabel('Channels')
# plt.ylabel('Channels')
# plt.show()

#%% EEG signal amplitude


# # Bandpass filter for the Delta band
# filtered_data = np.array([bandpass_filter(channel, *delta_band, sampling_rate) for channel in erp_after_alcohol])

# # Average across channels
# average_signal = np.mean(filtered_data, axis=0)

average_signal = np.mean(erp_before_alcohol, axis=0)

# Plotting the average EEG signal amplitude
plt.figure(figsize=(12, 6))
plt.plot(time, average_signal)
plt.title('Average EEG Signal Amplitude Over Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude (µV)')
plt.grid(True)
plt.show()

#%%

# # Plotting the average EEG signal amplitude in the Delta band
# plt.figure(figsize=(12, 6))
# plt.plot(time, average_signal)
# plt.title('Average EEG Signal Amplitude in Delta Band Over Time')
# plt.xlabel('Time (seconds)')
# plt.ylabel('Amplitude (µV)')
# plt.grid(True)
# plt.show()