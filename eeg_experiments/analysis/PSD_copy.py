#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 22:46:03 2023

@author: kaloso
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
import os
import mne
# from mne_connectivity import spectral_connectivity

#%%

epochs = mne.read_epochs('epoch_data6-epo.fif', preload=True)

data = epochs.get_data()[0,:, :]

data2 = epochs.get_data()[-1,:, :]

# fif_files = [f'epoch_data{i}-epo.fif' for i in range(1, 18)]

# # Load and extract the first epoch from each file
# first_epochs = []
# for file in fif_files:
#     epochs = mne.read_epochs(file, preload=True)
#     first_epoch = epochs[0].get_data()  # Extracts the first epoch
#     first_epochs.append(first_epoch)
#%% psd for epoch 1    


# # Parameters for Welch's method
# sfreq = 128  # Sampling frequency
# nperseg = 256  # Length of each segment for Welch's method

# # Function to calculate PSD using Welch's method
# def calculate_psd(epoch_data):
#     psds = np.zeros((epoch_data.shape[0], nperseg // 2 + 1))  # (n_channels, PSD length)
#     for i in range(epoch_data.shape[0]):  # Loop over channels
#         freqs, psd = welch(epoch_data[i, :], sfreq, nperseg=nperseg)
#         psds[i, :] = psd
#     return freqs, psds

# # Calculate PSD for the first epoch of each participant
# all_psds = []
# for epoch in first_epochs:
#     freqs, psds = calculate_psd(epoch[0])  # Assuming epoch shape is (1, n_channels, n_samples)
#     all_psds.append(psds)

# # all_psds is a list where each item contains the PSDs for all channels of a participant's first epoch

# average_psds = [np.mean(psds, axis=0) for psds in all_psds]  # Average across channels
    

# plt.figure(figsize=(12, 8))

# # Define maximum y-value for consistent y-axis limits, based on your data
# y_max = max([psd.max() for psd in average_psds]) * 1.1  # 10% more than the max value

# # Define a colormap to use for each participant's line
# colors = plt.cm.viridis(np.linspace(0, 1, len(average_psds)))

# for i, avg_psd in enumerate(average_psds):
#     plt.plot(freqs, avg_psd, label=f'Participant {i + 1}', color=colors[i])

# plt.title('Average Power Spectral Density Across All Channels for Each Participant')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Average Power Spectral Density (dB/Hz)')
# plt.yscale('log')  # Optional: Uncomment if a log scale is preferred
# plt.ylim(0, y_max)  # Set consistent y-axis limits
# plt.xlim(freqs[0], freqs[-1])  # Set x-axis limits based on your frequency range
# plt.grid(True)  # Add grid lines
# plt.legend(loc='upper right')
# plt.tight_layout()  # Adjust the layout
# plt.show()


# plt.figure(figsize=(12, 8))

# for i, avg_psd in enumerate(average_psds):
#     plt.plot(freqs, avg_psd, label=f'PSubject {i + 1}')

# plt.title('Average Power Spectral Density Across All Channels for Each Participant')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Average Power Spectral Density (dB/Hz)')
# plt.legend()
# plt.show()

#%% psd for epoch 6

# # Function to calculate PSD using Welch's method for available channels
# def calculate_psd_for_available_epochs(epoch_data):
#     # Check if the sixth epoch is available (epoch index 5)
#     if epoch_data.shape[2] > 5:
#         psds = []
#         for channel_data in epoch_data:
#             # Calculate PSD for each available channel in the sixth epoch
#             freqs, psd = welch(channel_data[:, 5], sfreq, nperseg=nperseg)
#             psds.append(psd)
#         return freqs, np.array(psds)
#     else:
#         return None, None

# # Calculate PSD for the sixth epoch of each participant
# all_psds_epoch_6 = []
# for epoch_data in first_epochs:
#     freqs, psds = calculate_psd_for_available_epochs(epoch_data)
#     if psds is not None:
#         all_psds_epoch_6.append(psds)
        
        
# for i, psds in enumerate(all_psds_epoch_6):
#     avg_psd = np.mean(psds, axis=0)  # Average across channels
#     plt.plot(freqs, avg_psd, label=f'Subject {i + 1}')

# # plt.title('Average Power Spectral Density Across Available Channels for Each Participant (Epoch 6)')
# # plt.xlabel('Frequency (Hz)')
# # plt.ylabel('Average Power Spectral Density (dB/Hz)')
# # plt.legend()
# # plt.show()  

# # Now we have all PSDs for epoch 6, let's plot them
# average_psds = [np.mean(psds, axis=0) for psds in all_psds_epoch_6]  # Average across channels for each participant

# # Assuming 'average_psds' and 'freqs' are already defined in your context
# num_subjects = len(average_psds)  # Determine the number of subjects

# # Choose a colormap that offers enough distinct colors for your subjects
# colormap = plt.cm.get_cmap('tab20', num_subjects)

# # Assign a color to each subject using the colormap
# colors = [colormap(i) for i in range(num_subjects)]

# # Improved plot code
# plt.figure(figsize=(12, 8))

# # Define maximum y-value for consistent y-axis limits, based on your data
# y_max = max([psd.max() for psd in average_psds]) * 1.1  # 10% more than the max value

# # Define a colormap to use for each participant's line
# colors = plt.cm.viridis(np.linspace(0, 1, len(average_psds)))

# for i, avg_psd in enumerate(average_psds):
#     plt.plot(freqs, avg_psd, label=f'Subject {i + 1}', color=colors[i])

# plt.title('Average Power Spectral Density Across All Channels for Each Participant (Epoch 6)')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Average Power Spectral Density (dB/Hz)')
# plt.ylim(0, y_max)  # Set consistent y-axis limits
# plt.xlim(freqs[0], freqs[-1])  # Set x-axis limits based on your frequency range
# plt.grid(True)  # Add grid lines
# plt.legend(loc='upper right')
# plt.tight_layout()  # Adjust the layout
# plt.show()


#%%      

# Sampling frequency
sfreq = epochs.info['sfreq']  # Get the sampling frequency from the epochs


# Define the Delta band frequency range
freq_band = (0.5, 4)

# Use Welch's method to compute the PSD for each channel
frequencies, psd = welch(data2, sfreq, nperseg=1024, axis=-1)

# Filter out frequencies outside the Delta band
delta_indices = np.where((frequencies >= freq_band[0]) & (frequencies <= freq_band[1]))
freqs_delta = frequencies[delta_indices]
psd_delta = psd[:, delta_indices[0]]

#%%


# Initialize variables to store the global min and max PSD values
global_psd_min, global_psd_max = np.inf, -np.inf

# # List of frequency bands to analyze
# freq_bands = {
#     'Delta': (0.5, 4),
#     'Theta': (4, 8),
#     'Alpha': (8, 12),
#     'Beta': (12, 25),
#     'Gamma': (25, 55)
# }

# # First pass: calculate PSD for each band and update global min/max
# for band, (low, high) in freq_bands.items():
#     # Assuming 'data' is your EEG data and 'sfreq' is the sampling frequency
#     delta_indices = np.where((frequencies >= low) & (frequencies <= high))
#     freqs_band = frequencies[delta_indices]
#     psd_band = psd[:, delta_indices[0]]

#     # Update global min/max
#     band_psd_min, band_psd_max = np.min(psd_band), np.max(psd_band)
#     global_psd_min = min(global_psd_min, band_psd_min)
#     global_psd_max = max(global_psd_max, band_psd_max)

#%%

# Plotting
plt.figure(figsize=(12, 8))
for i in range(psd_delta.shape[0]):
    plt.plot(freqs_delta, 10 * np.log10(psd_delta[i]), label=f'Channel {i + 1}')

plt.title('Band Power Spectral Density of Each Channel')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (dB/Hz)')
plt.ylim(10 * np.log10(global_psd_min), 10 * np.log10(global_psd_max))
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.tight_layout()
plt.show()