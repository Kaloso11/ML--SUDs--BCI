#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 16:15:14 2023

@author: kaloso
"""
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 
import seaborn as sns


#%%





# %%


# Number of components for PCA
n_components = 2  # Adjust based on your requirements

# # Subjects to exclude
# exclude_subs = [1, 12]

#%%

# # Load all sub data
# subs = [np.load(f'sub{i}epoch/sub{i}epoch6.npy') for i in range(1, 18)]

# # Exclude subjects 1 and 12 for epoch 6
# subs = [np.load(f'sub{i}epoch/sub{i}epoch6.npy') for i in range(2, 18) if i != 12]

# # Exclude subjects 1, 4, and 12 for epoch 6
# subs = [f'dwt{i}_testing.npy' for i in range(1, 18)]

# # Create a list to store PCA-transformed dataframes
# dfs = []

# # Perform PCA on each dataset and store in dataframe list
# for i, sub in enumerate(subs, start=1):
#     pca = PCA(n_components=n_components)
#     pca_transformed = pca.fit_transform(sub.T)  # Transpose the data for PCA
#     df = pd.DataFrame(pca_transformed, columns=[f'PCA1', f'PCA2'])
#     #df['Subject'] = f'Sub{i}'  # Add a column to indicate the subject number
#     dfs.append(df)

# # Concatenate all dataframes vertically
# final_df = pd.concat(dfs, ignore_index=True)

#print(final_df)

#%%

file_paths = [f'data{i}_testing.npy' for i in range(1, 18)]


# Store the PCA-transformed DataFrames
dfs = []

# Loop through each file, perform PCA, and store the results
for i, file_path in enumerate(file_paths):
    try:
        # Load the 3D array dataset
        data_3d = np.load(file_path)

        # Reshape the data by combining the 'n_channels' with 'n_epochs' as features
        # and keeping 'n_samples' as the number of observations
        n_channels, n_samples, n_epochs = data_3d.shape
        data_2d = data_3d.reshape(n_channels, n_samples * n_epochs).T

        # Perform PCA on the 2D data
        pca = PCA(n_components=n_components)
        pca_transformed = pca.fit_transform(data_2d)

        # Create a DataFrame with the PCA results
        df = pd.DataFrame(pca_transformed, columns=[f'PCA{i}' for i in range(1, n_components + 1)])

        # Add a column for the subject number
        # df['Subject'] = f'Sub{i+1}'

        # Append to the list of DataFrames
        dfs.append(df)

    except FileNotFoundError:
        print(f'File {file_path} not found. Skipping...')
    except Exception as e:
        print(f'An error occurred with file {file_path}: {e}')

# Combine all the individual DataFrames into one
final_df = pd.concat(dfs, ignore_index=True)



final_df.to_csv('final_test.csv', index=False)
