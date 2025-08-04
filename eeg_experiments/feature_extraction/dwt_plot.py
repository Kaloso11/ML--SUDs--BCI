#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 14:38:09 2023

@author: kaloso
"""

import numpy as np
import matplotlib.pyplot as plt
import mne


import warnings
mne.set_log_level("WARNING")
warnings.filterwarnings("ignore")

# Load DWT data for each participant
dwt_data = [np.load(f'dwt{i}.npy') for i in range(1, 18)]

# # Function to plot DWT coefficients for all channels and samples for a given epoch
# def plot_dwt_for_epoch(dwt_data, epoch_number):
#     plt.figure(figsize=(12, 8))

#     # Loop over each participant's DWT data
#     for participant_index, participant_dwt in enumerate(dwt_data):
#         # Check if the epoch number is within the range
#         if participant_dwt.shape[2] > epoch_number:
#             # Select the data for the given epoch
#             epoch_data = participant_dwt[:, :, epoch_number]

#             # Plot DWT coefficients for each channel
#             for channel_index in range(epoch_data.shape[0]):
#                 plt.plot(epoch_data[channel_index, :], label=f'Participant {participant_index + 1} - Channel {channel_index + 1}')

#     plt.title(f'DWT Coefficients Across All Channels for Each Participant (Epoch {epoch_number + 1})')
#     plt.xlabel('Samples')
#     plt.ylabel('DWT Coefficients')
#     plt.legend(loc='upper right')
#     plt.tight_layout()
#     plt.show()

# # Plot DWT for epoch 1 (index 0) and epoch 6 (index 5)
# plot_dwt_for_epoch(dwt_data, 0)  # For epoch 1
# plot_dwt_for_epoch(dwt_data, 5)  # For epoch 6


def plot_average_dwt(dwt_data, epoch_number):
    plt.figure(figsize=(12, 8))

    # Loop over each participant's DWT data
    for participant_dwt in dwt_data:
        # Skip if the specified epoch is not available
        if participant_dwt.shape[2] <= epoch_number:
            continue

        # Calculate the average DWT across all channels for the specified epoch
        avg_dwt = np.mean(participant_dwt[:, :, epoch_number], axis=0)
        plt.plot(avg_dwt, label=f'Participant {dwt_data.index(participant_dwt) + 1}')

    plt.title(f'Average DWT Coefficients Across All Channels (Epoch {epoch_number + 1})')
    plt.xlabel('Samples')
    plt.ylabel('Average DWT Coefficients')
    plt.legend()
    plt.show()

# Assuming dwt_data is a list of numpy arrays for each participant
# Each array has the shape (n_channels, n_samples, n_epochs)

# Load DWT data for each participant
dwt_files = [f'dwt{i}.npy' for i in range(1, 18)]
dwt_data = [np.load(file) for file in dwt_files]

# Plot the average DWT for all channels for epoch 1 and epoch 6
plot_average_dwt(dwt_data, 0)  # For epoch 1
plot_average_dwt(dwt_data, 5)  # For epoch 6
