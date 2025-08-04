#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 01:48:43 2023

@author: kaloso
"""

import numpy as np 
import mne
#from data_preprocessing import ica_data

import warnings
mne.set_log_level("WARNING")
warnings.filterwarnings("ignore")

# %% Function to perform batching
def perform_batching(data_batch, batch_size):
    n_channels, n_times, n_epochs = data_batch.shape

    batched_data = []
    for epoch_start in range(0, n_epochs, batch_size):
        epoch_end = min(epoch_start + batch_size, n_epochs)
        batch = data_batch[:, :, epoch_start:epoch_end]
        #reshaped_batch = batch.transpose(0, 2, 1)  # (channels, batch_size, n_times)
        batched_data.append(batch)

    return batched_data

if __name__ == "__main__":
    # Load your ICA-processed EEG data
    #data_batch = np.load('epoch_data17.fif') 
    data_batch = mne.read_epochs('epoch_data2_test.fif', preload=True)
    data_batch = data_batch.get_data()
    data_batch = np.transpose(data_batch,(1, 2, 0))
    batch_size = 32

    # Perform batching
    batched_data = perform_batching(data_batch, batch_size)
    
    batched_data = np.squeeze(batched_data)

    # Save the batched data
    # np.save('data17.npy', np.array(batched_data))
    # np.save('data16_testing.npy', np.array(batched_data))
    
    # load = np.load('batched_data2.npy',allow_pickle=True)
    
    # print(load.shape)

#%%