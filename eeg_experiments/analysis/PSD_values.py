#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 22:46:03 2023

@author: kaloso
"""


import pandas as pd
import numpy as np
from scipy.signal import welch
import mne

# Define the frequency bands
bands = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 12),
    'Beta': (12, 25),
    'Gamma': (25, 55)
}

def band_power(data, sf, band):
    freqs, psd = welch(data, sf, nperseg=1024)
    freq_range = np.where((freqs >= band[0]) & (freqs <= band[1]))[0]
    band_psd = psd[freq_range]
    return np.trapz(band_psd, freqs[freq_range])

# Process each subject file
all_subjects_df = []
for subject_id in range(1, 18):  # Assuming you have 17 subjects
    file_name = f'epoch_data{subject_id}-epo.fif'
    epochs = mne.read_epochs(file_name, preload=True)
    sfreq = epochs.info['sfreq']

    # Extract the first epoch
    first_epoch_data = epochs.get_data()[0, :, :]

    # Initialize DataFrame for the current subject
    num_channels = len(epochs.info['ch_names'])
    subject_df = pd.DataFrame(index=range(num_channels), columns=bands.keys())

    # Compute band power for each channel and band
    for i in range(num_channels):
        channel_data = first_epoch_data[i, :]  # Extract data for each channel
        for band_name, freq_range in bands.items():
            subject_df.at[i, band_name] = band_power(channel_data, sfreq, freq_range)

    # Set multi-index (Subject ID, Channel)
    subject_df.index = pd.MultiIndex.from_product([[subject_id], epochs.info['ch_names']], names=['Subject', 'Channel'])

    # Append to the list
    all_subjects_df.append(subject_df)

# Concatenate all subject dataframes
final_df = pd.concat(all_subjects_df)

# Display the DataFrame
# print(final_df)


# Select and print data for Subject 1
subject_1_data = final_df.loc[4]
print("Band Power Data for Subject 4:")
print(subject_1_data)