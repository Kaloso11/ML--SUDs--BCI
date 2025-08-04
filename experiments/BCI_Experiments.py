#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 09:45:51 2022

@author: kaloso
"""

import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
mne.set_log_level("WARNING")

#SUBJECT 1
SUD1 = os.path.join("Documents/MEng Research/BCI Experiments/Subject 1", "*.csv")
SUD1 = glob.glob(SUD1)
SUD1 = pd.concat(map(pd.read_csv, SUD1), ignore_index=True)

#SUBJECT 2
SUD2 = pd.read_csv("Documents/MEng Research/BCI Experiments/Subject 2/subject2_EPOCFLEX_161609_2022.06.22T11.39.33+02.00.md.bp.csv",sep=",")

#SUBJECT 3
SUD3 = os.path.join("Documents/MEng Research/BCI Experiments/Subject 3", "*.csv")
SUD3 = glob.glob(SUD3)
SUD3 = pd.concat(map(pd.read_csv, SUD3), ignore_index=True)

#SUBJECT 4
SUD4 = os.path.join("Documents/MEng Research/BCI Experiments/Subject 4", "*.csv")
SUD4 = glob.glob(SUD4)
SUD4 = pd.concat(map(pd.read_csv, SUD4), ignore_index=True)

#SUBJECT 5
SUD5 = pd.read_csv("Documents/MEng Research/BCI Experiments/Subject 5/SUBJECT 5_EPOCFLEX_161876_2022.06.24T14.42.02+02.00.bp.csv",sep=",")

#SUBJECT 6
SUD6 = pd.read_csv("Documents/MEng Research/BCI Experiments/Subject 6/Subject 6_EPOCFLEX_161875_2022.06.24T14.42.03+02.00.md.bp.csv",sep=",")

#SUBJECT 7
SUD7 = os.path.join("Documents/MEng Research/BCI Experiments/Subject 7", "*.csv")
SUD7 = glob.glob(SUD7)
SUD7 = pd.concat(map(pd.read_csv, SUD7), ignore_index=True)

#SUBJECT 8
SUD8 = pd.read_csv("Documents/MEng Research/BCI Experiments/Subject 8/Subject 8_EPOCFLEX_161937_2022.06.25T16.50.54+02.00.bp.csv",sep=",")

#SUBJECT 9
SUD9 = os.path.join("Documents/MEng Research/BCI Experiments/Subject 9", "*.csv")
SUD9 = glob.glob(SUD9)
SUD9 = pd.concat(map(pd.read_csv, SUD9), ignore_index=True)

#SUBJECT 10
SUD10 = pd.read_csv("Documents/MEng Research/BCI Experiments/Subject 10/Subject 10_EPOCFLEX_2022.07.05T12.20.01+02.00.bp.csv",sep=",")

#SUBJECT 11
SUD11 = os.path.join("Documents/MEng Research/BCI Experiments/Subject 11", "*.csv")
SUD11 = glob.glob(SUD11)
SUD11 = pd.concat(map(pd.read_csv, SUD11), ignore_index=True)

#SUBJECT 12
SUD12 = os.path.join("Documents/MEng Research/BCI Experiments/Subject 12", "*.csv")
SUD12 = glob.glob(SUD12)
SUD12 = pd.concat(map(pd.read_csv, SUD12), ignore_index=True)

#SUBJECT 13
SUD13 = pd.read_csv("Documents/MEng Research/BCI Experiments/Subject 13/Subject 13_EPOCFLEX_162505_2022.07.06T11.49.51+02.00.bp.csv",sep=",")

#SUBJECT 14
SUD14 = pd.read_csv("Documents/MEng Research/BCI Experiments/Subject 14/Subject 14_EPOCFLEX_162804_2022.07.11T11.31.35+02.00.md.bp.csv",sep=",")

#SUBJECT 15
SUD15 = pd.read_csv("Documents/MEng Research/BCI Experiments/Subject 15/Subject 15_EPOCFLEX_162805_2022.07.11T11.31.40+02.00.bp.csv",sep=",")

#SUBJECT 16
SUD16 = pd.read_csv("Documents/MEng Research/BCI Experiments/Subject 16/Subject 16_EPOCFLEX_162815_2022.07.11T14.18.42+02.00.md.bp.csv",sep=",")

#SUBJECT 17
SUD17 = os.path.join("Documents/MEng Research/BCI Experiments/Subject 17", "*.csv")
SUD17 = glob.glob(SUD17)
SUD17 = pd.concat(map(pd.read_csv, SUD12), ignore_index=True)
