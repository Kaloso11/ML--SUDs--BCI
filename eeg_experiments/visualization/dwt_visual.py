#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 01:18:28 2023

@author: kaloso
"""

import pywt
import matplotlib.pyplot as plt
import numpy as np


#%%
# # Parameters
# num_subjects = 17
# num_epochs = 6

# # Assuming maximum number of channels across all subjects for memory allocation
# max_channels = 16  # Modify this if some subjects have more than 16 channels

# # Create a placeholder for aggregated coefficients
# aggregate_coeff = np.zeros((max_channels, 153601, num_epochs))

# # Loop through subjects and epochs
# for subject in range(1, num_subjects + 1):
#     for epoch in range(1, num_epochs + 1):
#         # Exclude subjects 1, 4, and 12 for epoch 6
#         if epoch == 6 and subject in [1, 4, 12]:
#             continue

#         coeff = np.load(f'sub{subject}epoch/sub{subject}epoch{epoch}.npy')
        
#         # Incrementally aggregate coefficients for averaging later
#         aggregate_coeff[:coeff.shape[0], :, epoch-1] += coeff

# # Average the coefficients (You can change this depending on what you're aiming for)
# aggregate_coeff /= num_subjects

# # Visualization
# for channel in range(max_channels):
#     plt.figure(figsize=(15, 5))
#     for epoch in range(num_epochs):
#         plt.plot(aggregate_coeff[channel, :, epoch], label=f'Epoch {epoch + 1}')
#     plt.title(f'Channel {channel + 1}')
#     plt.legend()
#     plt.show()

#%%

# Parameters
num_subjects = 17
num_epochs = 6

all_data = []

# Loop through subjects
for subject in range(1, num_subjects + 1):
    subject_data = []
    
    for epoch in range(1, num_epochs + 1):
        # Exclude subjects 1, 4, and 12 for epoch 6
        if epoch == 6 and subject in [1, 4, 7, 12]:
            continue

        coeff = np.load(f'sub{subject}epoch/sub{subject}epoch{epoch}.npy')
        subject_data.append(coeff)
        
    all_data.append(subject_data)

# # Visualization
# for epoch in range(num_epochs):
#     plt.figure(figsize=(15, 5))
#     for subject in range(num_subjects):
#         # We'll plot the average of each channel for each subject
#         # If subjects have different channel numbers, this will only take into account the minimum common channels
#         avg_coeff = all_data[subject][epoch].mean(axis=0)
#         plt.plot(avg_coeff, label=f'Subject {subject + 1}')
#     plt.title(f'Epoch {epoch + 1}')
#     plt.legend()
#     plt.show()


# Visualization
for epoch in range(num_epochs):
    plt.figure(figsize=(15, 5))
    
    for subject in range(num_subjects):
        # Check if the epoch exists for the given subject
        if epoch < len(all_data[subject]):
            # We'll plot the average of each channel for each subject
            avg_coeff = all_data[subject][epoch].mean(axis=0)
            plt.plot(avg_coeff, label=f'Subject {subject + 1}')

    plt.title(f'Epoch {epoch + 1}')
    plt.legend()
    plt.show()
