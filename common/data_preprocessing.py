#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 20:31:57 2022

@author: kaloso
"""

import mne
import numpy as np
import BCI_Data_17 as s17


import warnings
mne.set_log_level("WARNING")
warnings.filterwarnings("ignore")


# %% IMPORT DATA
SUD17 = s17.SUD17

# print(SUD17)

#DROP EXCESS CHANNELS TO RETAIN 32 EEG CHANNELS

SUD17.drop(SUD17.columns[0:3],axis=1,inplace=True)
SUD17.drop(SUD17.columns[32:],axis=1,inplace=True)


#DEFINE CHANNELS NAMES

ch_names = ['Cz', 'Fz','Fp1', 'F7', 'F3', 'FC1', 'C3', 'FC5', 'FT9', 'T7', 'CP5', 'CP1', 'P3', 
            'P7', 'PO9','O1', 'Pz',	'Oz', 'O2',	'PO10',	
            'P8', 'P4',	'CP2',	'CP6', 'T8', 'FT10', 'FC6',	'C4', 'FC2','F4', 'F8', 'Fp2']

sfreq = 128.0

info = mne.create_info(ch_names, sfreq,ch_types='eeg',verbose=None)

samples = np.array(SUD17)
#samples = cuda.to_device(samples1)

data = samples.T

subject = mne.io.RawArray(data,info)
#subject2 = mne.io.RawArray(data,info)

#subject.plot

# montage_kind = "standard_1020"
# montage =  mne.channels.make_standard_montage(montage_kind)
# subject.set_montage(montage, match_case=False)

# %% CREATE EVENTS 


picks=mne.pick_channels(ch_names,include=[],exclude=['bads'])

#picks=mne.pick_types(subject1.info,exclude='bads')

events = mne.make_fixed_length_events(subject,id=1,start=0,stop=None,duration=1200,first_samp=True)


# %% FILTERING THE DATA and EPOCHING

# subject.notch_filter(np.arange(60,120,60))

# # filtered_data = eeg.filt_data(subject1, 1.0, 55, sfreq)

# filtered_data = subject.copy().filter(0.5, 55)
#filtered_data = subject1.filter_data(1.0, 55)

#raw_data = filtered_data.get_data()

#subject1.plot_psd()
epochs = mne.Epochs(subject, events, event_id=1, tmin=0, tmax=1200,picks=picks, baseline=(None, None), reject=None, preload=True,reject_by_annotation=True)
#df_epoch = epochs.to_data_frame()
#epochs.plot()

# # Compute TFR using Morlet wavelets
# fmin, fmax, freqs = 2, 60, np.arange(2., 60., 3.)
# power = mne.time_frequency.tfr_multitaper(epochs, freqs=freqs, n_cycles=freqs / 2, time_bandwidth=2.0,
#                                           return_itc=False)
# power.plot([0], baseline=(-0.5, 0), mode='logratio', title=power.ch_names[0])

# if __name__ == "__main__":
    
#     epochs.save('epoch_data16_test.fif', overwrite=True)
    
    # # Load the saved 'new_data.npy'
    # e_data = np.load('epoch_data1.fif')

    # # Check the shape of the loaded data
    # print("Shape of epoched_data:", e_data.shape)

# %%

