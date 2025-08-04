#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 01:49:01 2023

@author: kaloso
"""

import mne
import numpy as np
import matplotlib.pyplot as plt

#%%

# Load ICA activations for all subjects
all_ica_data = [np.load(f'ica_data{i}.npy') for i in range(1, 18)]

# List of epoch files for each subject
epoch_files = [f'epoch_data{i}.fif' for i in range(1, 18)]


#%%

# Iterate over epochs and plot topographies
for epoch_num in range(6):  # As we have 6 epochs max
    combined_ica_activation = []

    # Load each participant's epoch data and ICA activation
    for idx, epoch_file in enumerate(epoch_files):
        epochs = mne.read_epochs(epoch_file, preload=True)
        info = epochs.info
        info = mne.pick_info(info, mne.pick_types(info, meg=False, eeg=True))

        # Check if this subject has this epoch
        if all_ica_data[idx].shape[2] > epoch_num:
            ica_activation = all_ica_data[idx][:, :, epoch_num]
            avg_ica_activation = np.mean(ica_activation, axis=1)
            combined_ica_activation.append(avg_ica_activation)

    # Compute average ICA activation across all subjects for this epoch
    avg_combined_ica_activation = np.mean(combined_ica_activation, axis=0)

    mne.viz.plot_topomap(avg_combined_ica_activation, info, cmap='jet')
    plt.title(f"Epoch {epoch_num + 1} - Average Topography")
    plt.show()

#%%