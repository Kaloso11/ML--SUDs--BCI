#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 00:40:34 2023

@author: kaloso
"""

import mne
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import numpy as np


#%%


# Load the epochs
epochs = mne.read_epochs('epoch_data2-epo.fif', preload=True)

# Get data for the first and last epoch
data = epochs.get_data()[0, :, :]  # First epoch
data2 = epochs.get_data()[-1, :, :]  # Last epoch

# Sampling frequency
sfreq = epochs.info['sfreq']  # Get the sampling frequency from the epochs

# Number of samples
n_samples = data.shape[1]

# Perform FFT on the data
fft_data = np.fft.rfft(data, axis=1)
freqs = np.fft.rfftfreq(n_samples, 1/sfreq)

# Define the frequency band range
freq_band = (0.5, 4)

#%%

# # Time vector
# times = np.arange(data.shape[1]) / sfreq

# # Plotting
# plt.figure(figsize=(12, 6))

# # Plotting each channel
# for channel_index in range(data.shape[0]):
#     plt.plot(times, data[channel_index, :], label=f'Channel {channel_index + 1}')

# plt.title('Signal Amplitude')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Moving the legend outside the plot

# plt.tight_layout()
# plt.show()

#%%

# Extract amplitudes within the Delta band
freq_indices = np.where((freqs >= freq_band[0]) & (freqs <= freq_band[1]))
freq_amplitudes = np.abs(fft_data[:, freq_indices[0]])

# # Plotting
# plt.figure(figsize=(12, 6))

# # Plot the average delta band amplitude across channels
# plt.plot(freqs[freq_indices], np.mean(freq_amplitudes, axis=0), label='Frequency Band')

# plt.title('Average Signal Amplitude - Epoch 1')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Amplitude')
# plt.legend()

# plt.tight_layout()
# plt.show()


# Plotting the average EEG signal amplitude
plt.figure(figsize=(12, 6))
# Plot the average delta band amplitude across channels
plt.plot(freqs[freq_indices], np.mean(freq_amplitudes, axis=0), label='Frequency Band')
plt.title('Average EEG Signal Amplitude Over Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude (µV)')
plt.grid(True)
plt.show()