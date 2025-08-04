#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 22:46:03 2023

@author: kaloso
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

sfreq = 128  # Sampling frequency
freq_band = (0.5, 4)  # Define the frequency band of interest

def plot_avg_psd_per_epoch_for_each_subject(epoch_number):
    # Initialize a list to store average PSD for each subject in the current epoch
    avg_psd_per_subject = []

    # Load and process data for each subject
    for i in range(1, 18):  # Assuming 17 subjects
        try:
            # Load data for the current subject and epoch
            data_subject = np.load(f'sub{i}epoch/sub{i}epoch{epoch_number}.npy')
            
            # Use Welch's method to compute the PSD for the subject's data
            frequencies, psd = welch(data_subject, sfreq, nperseg=1024, axis=-1)
            
            # Filter out frequencies outside the specified band
            delta_indices = np.where((frequencies >= freq_band[0]) & (frequencies <= freq_band[1]))
            freqs_delta = frequencies[delta_indices]
            psd_delta = psd[:, delta_indices[0]]
            
            # Calculate the average PSD for this subject in the current epoch
            avg_psd_subject = np.mean(psd_delta, axis=0)
            avg_psd_per_subject.append(avg_psd_subject)
            
        except FileNotFoundError:
            #print(f"Data for Subject {i}, Epoch {epoch_number} not found.")
            continue
    
    # Calculate the average PSD across all subjects for this epoch
    avg_psd_across_subjects = np.mean(np.array(avg_psd_per_subject), axis=0)
    
    # Plotting
    plt.figure(figsize=(12, 8))
    plt.plot(freqs_delta, 10 * np.log10(avg_psd_across_subjects), label=f'Epoch {epoch_number}')
    plt.title(f'Average Delta Band PSD Across Subjects - Epoch {epoch_number}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (dB/Hz)')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Call the function for each epoch you wish to process
for epoch in range(1, 7):  # Adjust the range as needed for your epochs
    plot_avg_psd_per_epoch_for_each_subject(epoch)

# plt.show()

#%%