# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 22:07:01 2022

@author: TM14001854
"""

import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from mne.preprocessing import ICA
mne.set_log_level("WARNING")


#SUBJECT 1
SUD1 = os.path.join("D:\Documents\MEng Research\BCI Experiments\Subject 1", "*.csv")
SUD1 = glob.glob(SUD1)
SUD1 = pd.concat(map(pd.read_csv, SUD1), ignore_index=True)

#SUBJECT 2
# SUD2 = pd.read_csv("D:\Documents\MEng Research\BCI Experiments\Subject 2\subject2_EPOCFLEX_161609_2022.06.22T11.39.33+02.00.md.bp.csv",sep=",")

#DROP EXCESS CHANNELS TO RETAIN 32 EEG CHANNELS

SUD1.drop(SUD1.columns[0:3],axis=1,inplace=True)
SUD1.drop(SUD1.columns[32:],axis=1,inplace=True)

#DEFINE CHANNELS NAMES

ch_names = ['Cz', 'Fz','Fp1', 'F7', 'F3', 'FC1', 'C3', 'FC5', 'FT9', 'T7', 'CP5', 'CP1', 'P3', 
            'P7', 'PO9',	'O1', 'Pz',	'Oz', 'O2',	'PO10',	
            'P8', 'P4',	'CP2',	'CP6', 'T8', 'FT10', 'FC6',	'C4', 'FC2','F4', 'F8', 'Fp2']

sfreq = 128.0

info = mne.create_info(ch_names, sfreq,ch_types='eeg',verbose=None)

samples = np.array(SUD1)

data = samples.T

subject1 = mne.io.RawArray(data,info)

montage_kind = "standard_1020"
montage =  mne.channels.make_standard_montage(montage_kind)
subject1.set_montage(montage, match_case=False)

#print(subject1.info)

#picks = mne.pick_types(subject1.info,exclude='bads')

picks=mne.pick_channels(ch_names,include=[],exclude=['bads'])

events = mne.make_fixed_length_events(subject1,id=1,start=5,stop=50,duration=2.5,first_samp=True)

epochs = mne.Epochs(subject1, events, event_id=1, tmin=0, tmax=2.5,picks=picks, baseline=(0, 0), reject=None, preload=True)

subject1 = subject1.notch_filter(np.arange(60,120,60))

subject1_new =subject1.copy().filter(0.1, 60)

