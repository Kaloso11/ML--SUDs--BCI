#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 15:51:49 2023

@author: kaloso
"""
import mne
import numpy as np
from mne.preprocessing import ICA
from autoreject import get_rejection_threshold

import warnings
mne.set_log_level("WARNING")
warnings.filterwarnings("ignore")

#%%   #INDEPENDENT COMPONENTS ANALYSIS


epoch_data = mne.read_epochs('epoch_data2-epo.fif', preload=True)

# data = epoch_data.get_data()[0,:, :]

reject = get_rejection_threshold(epoch_data)
#print(df_epoch.columns)

num_components = 0.99

ica = ICA(n_components=num_components, method='fastica',random_state=42)

ica.fit(epoch_data,reject=reject,tstep=600)

# pc = ica.apply(epoch_data)

ica_data = ica.get_sources(epoch_data).get_data()

ica_data = np.transpose(ica_data,(1, 2, 0))

# if __name__ == "__main__":
    
    # ica.save('epoch_data2_test.fif', overwrite=True)
    
    # # Load the saved 'new_data.npy'
    # loaded_data = np.load('ica_data2.npy')

    # # Check the shape of the loaded data
    # print("Shape of loaded_data:", loaded_data.shape)
    
    
# %%    


