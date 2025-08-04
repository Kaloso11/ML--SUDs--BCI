#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 14:24:15 2023

@author: kaloso
"""


from sklearn.cluster import KMeans, DBSCAN, OPTICS, Birch
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import mne 
# import hdbscan
from sklearn.neighbors import NearestNeighbors
# from tqdm import tqdm
# import cuml
# from cuml.cluster import DBSCAN, KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# import Pool
# import cudf


import warnings
mne.set_log_level("WARNING")
warnings.filterwarnings("ignore")

#%%


# pca_df = pd.read_csv('pca_df.csv')

# final_df = pd.read_csv("Documents/MEng_Research/pca_data/pca_df1.csv")

# final_df = np.load(f'dwt.npy')

# dwt_data = [np.load(f'dwt{i}.npy') for i in range(1, 18)]

# data_df = [np.load(f'sub{i}epoch/sub{i}epoch1.npy') for i in range(1,18)]

# final_df = np.load(f'sub6epoch/sub6epoch1.npy')

# final_df = final_df.T

# pca = PCA(n_components=2,random_state=123)
# principal_comp = pca.fit_transform(final_df)
# pca_df = pd.DataFrame(data=principal_comp,columns=['pca1','pca2'])

pca_df = pd.read_csv('data_pca.csv')

# pca_df = pd.read_csv('pca_6_alpha.csv')


#%% K-MEANS CLUSTERING

#ELBOW METHOD

# ks = range(1, 10)
# inertias = []

# for k in ks:
#     model = KMeans(n_clusters=k)
#     model.fit(pca_df)
#     inertias.append(model.inertia_)

# plt.figure(figsize=(8,5))
# plt.style.use('bmh')
# plt.plot(ks, inertias, '-o')
# plt.xlabel('Number of clusters, k')
# plt.ylabel('Inertia')
# plt.xticks(ks)
# plt.show()


# def K_means(data_frame):

#     k_means = KMeans(n_clusters = 3, random_state=123, n_init=30)
#     global me
#     me=k_means.fit(data_frame)
#     global c_labels
#     c_labels = k_means.labels_
    
    
# k_cluster=K_means(data_frame=pca_df)

# arr_k = np.unique(c_labels)
# #print(arr_k) 
# #print(c_labels[:1000])
# pca_dfk2 = pd.concat([pca_df,pd.DataFrame({'cluster':c_labels})],axis=1)

# fig = plt.figure(figsize=(12,8))
# plt.title('All_subjects : K_means Epoch6')
# ax = sns.scatterplot(x="PCA1",y="PCA2",hue='cluster',data=pca_dfk2,palette=['red','green','blue'])#,'yellow','purple','black','pink','orange','maroon'])
# plt.ylim(-80, 80)
# plt.show()


#%%

# # Iterate over different values of threshold and branching factor
# for threshold in [0.5, 1.0, 1.5]:  # Adjust as needed
#     for branching_factor in [20, 50, 100, 500]:  # Adjust as needed
#         clusterer = Birch(threshold=threshold, branching_factor=branching_factor, n_clusters=None)
#         clusterer.fit(pca_df)
#         labels = clusterer.labels_
        
#         n_clusters = len(set(labels))
        
#         print(f"threshold: {threshold}, branching_factor: {branching_factor} => Clusters: {n_clusters}")


# # Iterate over a broader range of threshold and branching factor values
# for threshold in [2.0, 2.5, 3.0, 3.5, 4.0]:
#     for branching_factor in [150, 200, 250, 300, 350, 400, 450, 500]:
#         clusterer = Birch(threshold=threshold, branching_factor=branching_factor, n_clusters=None)
#         clusterer.fit(pca_df)
#         labels = clusterer.labels_
        
#         n_clusters = len(set(labels))
        
#         print(f"threshold: {threshold}, branching_factor: {branching_factor} => Clusters: {n_clusters}")

#%%

def birch(data_frame):


    br_hierarchy = Birch(branching_factor = 500, n_clusters = 3, threshold = 1.5)
    global bir
    bir=br_hierarchy.fit(data_frame)
    global b_labels
    b_labels = br_hierarchy.labels_
    #h_clust = hierarchy.predict(data_frame)
    
b_cluster = birch(data_frame=pca_df)

# arr_b = np.unique(b_labels)

arr_b = np.bincount(b_labels)

#print(arr_b)
pca_dfb2 = pd.concat([pca_df,pd.DataFrame({'cluster':b_labels})],axis=1)

fig = plt.figure(figsize=(12,8))
plt.title('All_subjects : BIRCH Clustering')
ax = sns.scatterplot(x="pca1",y="pca2",hue='cluster',data=pca_dfb2,palette=['red','green','blue'])#,'yellow','purple','black','pink','orange','maroon'])
plt.ylim(-50, 60)
plt.xlim(-50, 60)
plt.show()


#%%