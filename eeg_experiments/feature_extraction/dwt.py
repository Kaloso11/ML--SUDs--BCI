#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 16:20:27 2023

@author: kaloso
"""

import numpy as np
import pywt
import matplotlib.pyplot as plt

#%%

eeg_data = np.load('dwt2.npy')

# eeg_data = raw_data


# # Assuming you have 'ica_data' as your ICA-processed EEG data (n_epochs, n_channels, n_samples)
n_channels, n_samples, n_epochs = eeg_data.shape


wavelet = 'haar'
level = 1

def compute_dwt_coeffs(eeg_data, wavelet='haar', level=level):
    """
    Compute DWT coefficients for EEG data.

    Parameters:
        eeg_data (numpy.ndarray): 3D EEG data array (n_channels, n_samples, n_epochs).
        wavelet (str): Wavelet type to use (e.g., 'haar', 'db1', 'db2', etc.).
        level (int): Level of decomposition for DWT.

    Returns:
        numpy.ndarray: DWT coefficients array of the same shape as eeg_data.
    """
    n_channels, n_samples, n_epochs = eeg_data.shape
    dwt_coeffs = np.zeros_like(eeg_data)  # Initialize an empty array to store DWT coefficients

    for epoch in range(n_epochs):
        for channel in range(n_channels):
            # Extract the time series for the current channel and epoch
            signal = eeg_data[channel, :, epoch]

            # Perform the DWT for the current channel and epoch
            coeffs = pywt.wavedec(signal, wavelet, level=level)

            # Store the DWT coefficients back in the array
            dwt_coeffs[channel, :, epoch] = np.concatenate(coeffs)[:n_samples]

    return dwt_coeffs


# Call the function to compute DWT coefficients
dwt_coeffs = compute_dwt_coeffs(eeg_data, wavelet=wavelet, level=level)

# if __name__ == "__main__":
    
#     np.save('dwt16_testing.npy', dwt_coeffs)


# # Choose a specific epoch for visualization
# epoch_idx = 1

# # Extract DWT coefficients for the selected epoch for all channels
# dwt_coeffs_epoch = dwt_coeffs[:, :, epoch_idx]

# # Calculate the half-point of the sample length to separate detail from approximation coefficients
# half_point = dwt_coeffs_epoch.shape[1] // 2

# # Plot the DWT coefficients (only detail coefficients) for all channels in the same epoch
# plt.figure(figsize=(10, 6))
# for channel_idx in range(dwt_coeffs.shape[0]):
#     plt.plot(dwt_coeffs_epoch[channel_idx, :half_point], label=f'Channel {channel_idx}')  # Only plotting up to the half_point
# plt.title(f'DWT Detail Coefficients for Epoch {epoch_idx}')
# plt.xlabel('Sample Index')
# plt.ylabel('Coefficient Value')

# # Move the legend outside of the plot
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# plt.show()

