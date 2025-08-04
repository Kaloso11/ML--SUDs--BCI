#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 22:46:03 2023

@author: kaloso
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
import os
import mne
from scipy.signal import butter, sosfiltfilt
from sklearn.decomposition import PCA
# from mne_connectivity import spectral_connectivity

#%%

# epochs = mne.read_epochs('epoch_data17-epo.fif', preload=True)

# data = epochs.get_data()[0,:, :]

# data2 = epochs.get_data()[-1,:, :]

# data = [np.load(f'sub{i}epoch/sub{i}epoch5.npy') for i in range(1,18)]

# data = []
# for i in range(1, 18):
#     try:
#         # Attempt to load the file
#         subject_data = np.load(f'sub{i}epoch/sub{i}epoch6.npy')
#         data.append(subject_data)
#     except FileNotFoundError:
#         # If the file doesn't exist, print a message and skip this subject
#         # print(f'Epoch 6 for subject {i} not found, skipping.')
#         continue

# combined_data = np.vstack(data)

#%%

data = np.load(f'sub9epoch1.npy')

data2 = np.load(f'sub9epoch6.npy')

# data2 = [np.load(f'sub{i}epoch/sub{i}epoch6.npy') for i in range(1,18)]

#%%      

# Sampling frequency
# sfreq = epochs.info['sfreq']  # Get the sampling frequency from the epochs

sfreq = 128

#%%

# # Define the Delta band frequency range
# freq_band = (4, 8)

# # Use Welch's method to compute the PSD for each channel
# frequencies, psd = welch(data2, sfreq, nperseg=1024, axis=-1)

# # Filter out frequencies outside the Delta band
# delta_indices = np.where((frequencies >= freq_band[0]) & (frequencies <= freq_band[1]))

# freqs_delta = frequencies[delta_indices]

# psd_delta = psd[:, delta_indices[0]]

# psd_band = psd[:, delta_indices[0]]

# # Calculate the average PSD across all subjects for this epoch
# avg_psd_delta = np.mean(psd_delta, axis=0)

# # Calculate the average PSD across all channels and all frequencies within this band
# avg_psd_band_all_channels = np.mean(psd_band)

# print (avg_psd_band_all_channels)

#%%

# plt.figure(figsize=(12, 8))
# for i in range(psd_delta.shape[0]):
#     # Correctly plotting the PSD for each channel in the Delta band
#     plt.plot(freqs_delta, 10 * np.log10(psd_delta[i, :]), label=f'Channel {i + 1}')

# plt.title('Beta Band PSD Per Channel - Subject8 Epoch6')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Power Spectral Density (dB/Hz)')
# plt.ylim(-43, 0)
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
# plt.tight_layout()
# plt.show()



#%%
# # Plotting
# plt.figure(figsize=(12, 8))
# for i in range(psd_delta.shape[0]):
#     avg_psd_per_channel = 10 * np.log10(np.mean(psd_delta[i, :], axis=-1))
#     plt.plot(freqs_delta, 10 * np.log10(avg_psd_delta[i, :]), label=f'Channel {i + 1}')
#     # plt.plot(freqs_delta, 10 * np.log10(avg_psd_delta), label='Average')


# plt.title('Average Delta Band PSD Across Subject3 - Epoch1')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Power Spectral Density (dB/Hz)')
# # plt.ylim(10 * np.log10(global_psd_min), 10 * np.log10(global_psd_max))
# plt.ylim(-55, 30)
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
# # plt.grid(True)
# plt.tight_layout()
# plt.show()

#%%

# # Function to apply a bandpass filter
# def bandpass_filter(combined_data, sfreq, freq_band):
#     sos = butter(N=4, Wn=freq_band, btype='bandpass', fs=sfreq, output='sos')
#     filtered_data = sosfiltfilt(sos, combined_data, axis=-1)  # Assuming time is the last axis
#     return filtered_data

# # Apply the filter to all epoch data for all participants
# filtered_data = bandpass_filter(combined_data, sfreq, freq_band)

# filtered_data = filtered_data.T

#%%

# # Initialize PCA, choose the number of components you wish to keep
# pca = PCA(n_components=2)  # Adjust n_components as needed

# # Fit PCA on the combined, filtered data
# pca_data = pca.fit_transform(filtered_data)

#%%

# # Convert the PCA data to a pandas DataFrame
# pca_df = pd.DataFrame(pca_data, columns=['PCA1', 'PCA2'])  # Assuming you kept 2 components

# # Save the DataFrame to a CSV file
# pca_df.to_csv('pca_6_alpha.csv')

#%%

# Define the Theta band frequency range
freq_band = (12, 35)

# Use Welch's method to compute the PSD for each channel
frequencies, psd = welch(data2, sfreq, nperseg=1024, axis=-1)

# Filter out frequencies outside the Theta band
theta_indices = np.where((frequencies >= freq_band[0]) & (frequencies <= freq_band[1]))

freqs_theta = frequencies[theta_indices]

psd_theta = psd[:, theta_indices[0]]

# Calculate the average PSD across all subjects for this epoch
avg_psd_theta = np.mean(psd_theta, axis=0)

# Calculate the average PSD across all channels and all frequencies within this band
avg_psd_band_all_channels = np.mean(psd_theta)

# print(f'Average PSD in the Theta band across all channels: {avg_psd_band_all_channels} μV²/Hz')

# Plotting the graph in μV²/Hz instead of dB/Hz
plt.figure(figsize=(12, 8))
for i in range(psd_theta.shape[0]):
    # Correctly plotting the PSD for each channel in the Theta band without converting to dB
    plt.plot(freqs_theta, psd_theta[i, :], label=f'Ch {i + 1}')

plt.title('Beta Band PSD Per Channel - Low risk user Epoch6',fontsize=20)
plt.xlabel('Frequency (Hz)',fontsize=20)
plt.ylabel('Power Spectral Density (μV²/Hz)',fontsize=20)
# Set the y-limit according to your PSD values range
plt.ylim(ymin=0)  # Replace with your specific range if needed
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
# plt.savefig("sub6_1_theta.eps", format='eps',dpi=300)
plt.tight_layout()
plt.show()


#%% Plot topography maps


