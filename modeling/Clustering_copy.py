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
import hdbscan
from sklearn.neighbors import NearestNeighbors
# from tqdm import tqdm
# import cuml
# from cuml.cluster import DBSCAN, KMeans
import matplotlib.pyplot as plt
# import Pool
# import cudf


import warnings
mne.set_log_level("WARNING")
warnings.filterwarnings("ignore")

#%%


# pca_df = pd.read_csv('pca_df.csv')

# # #SUBJECT 1
# df = os.path.join("Documents/MEng_Research/pca_data", "*.csv")
# df = glob.glob(df)
# df = pd.concat(map(pd.read_csv, df), ignore_index=True)

final_df = pd.read_csv("Documents/MEng_Research/pca_data/pca_df1.csv")


#df = df.iloc[0:15000000,:]

# df.to_csv('data_pca.csv', index=False)

# data_pca = pd.read_csv('Documents/MEng_Research/pca_data/data_pca.csv')

# data_pca = pd.read_csv('data_pca.csv')

# sample_data = df.sample(n=100000, random_state=42)

# # Convert your dataframe to a GPU dataframe
# data_gpu = cudf.DataFrame.from_pandas(data_pca)

#%% K-MEANS CLUSTERING

#ELBOW METHOD

# ks = range(1, 10)
# inertias = []

# for k in ks:
#     model = KMeans(n_clusters=k)
#     model.fit(final_df)
#     inertias.append(model.inertia_)

# plt.figure(figsize=(8,5))
# plt.style.use('bmh')
# plt.plot(ks, inertias, '-o')
# plt.xlabel('Number of clusters, k')
# plt.ylabel('Inertia')
# plt.xticks(ks)
# plt.show()


def K_means(data_frame):

    k_means = KMeans(n_clusters = 4, random_state=123, n_init=30)
    global me
    me=k_means.fit(data_frame)
    global c_labels
    c_labels = k_means.labels_
    
    
k_cluster=K_means(data_frame=final_df)

arr_k = np.unique(c_labels)
#print(arr_k) 
#print(c_labels[:1000])
pca_dfk2 = pd.concat([final_df,pd.DataFrame({'cluster':c_labels})],axis=1)

fig = plt.figure(figsize=(12,8))
ax = sns.scatterplot(x="pca1",y="pca2",hue='cluster',data=pca_dfk2,palette=['red','green','blue','yellow'])#,'purple','black','pink','orange','maroon'])
plt.show()



#%% DBSCAN


# def Dbscan(data_frame):

#     dbscan = DBSCAN(eps=0.5,min_samples=5,metric='euclidean')
#     global db
#     db=dbscan.fit(data_frame)
#     global d_labels
#     d_labels = dbscan.labels_
#     #d_clust = dbscan.predict(data_frame)
    
# d_cluster = Dbscan(data_frame=final_df)
# arr_d = np.unique(d_labels)
# print(arr_d)


# pca_dfd2 = pd.concat([final_df,pd.DataFrame({'cluster':d_labels})],axis=1)
# fig = plt.figure(figsize=(12,8))
# ax = sns.scatterplot(x="pca1",y="pca2",hue='cluster',data=pca_dfd2)#,palette=['red','green','blue','yellow','purple','black'])#,'pink','orange','maroon'])
# plt.show()


# # Iterate over different values of eps (radius of neighborhood) and min_samples
# for eps in [0.5, 1.0, 1.5]:  # these are common starting points for eps, but adjust as necessary
#     for min_s in [5, 10, 15]:
#         clusterer = DBSCAN(eps=eps, min_samples=min_s)
#         labels = clusterer.fit_predict(final_df)
        
#         n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # -1 indicates noise in DBSCAN
#         n_noise = list(labels).count(-1)
        
#         print(f"eps: {eps}, min_samples: {min_s} => Clusters: {n_clusters}, Noise points: {n_noise}")


#%% HDBSCAN

# # Initialize the HDBSCAN clusterer
# clusterer = hdbscan.HDBSCAN(min_samples=5, min_cluster_size=500)

# clusterer.fit(final_df)

# # Fit the model
# labels = clusterer.labels_
# # Do something with the labels...
# #print(labels)

# pca_dfd2 = pd.concat([final_df,pd.DataFrame({'cluster':labels})],axis=1)
# fig = plt.figure(figsize=(12,8))
# ax = sns.scatterplot(x="PCA1",y="PCA2",hue='cluster',data=pca_dfd2,palette = {-1: 'green', 0: 'blue', 1: 'red', 2: 'purple'})#['red','green','blue'],'yellow','purple','black'],'pink','orange','maroon'])
# plt.show()


# # Iterate over different values of min_samples and min_cluster_size
# for min_s in [5, 10, 15]:
#     for min_c in [100, 500, 1000]:
#         clusterer = hdbscan.HDBSCAN(min_samples=min_s, min_cluster_size=min_c)
#         clusterer.fit(final_df)
#         labels = clusterer.labels_
#         n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
#         n_noise = list(labels).count(-1)
#         print(f"min_samples: {min_s}, min_cluster_size: {min_c} => Clusters: {n_clusters}, Noise points: {n_noise}")

#%%

# # Iterate over different values of threshold and branching factor
# for threshold in [0.5, 1.0, 1.5]:  # Adjust as needed
#     for branching_factor in [20, 50, 100]:  # Adjust as needed
#         clusterer = Birch(threshold=threshold, branching_factor=branching_factor, n_clusters=None)
#         clusterer.fit(final_df)
#         labels = clusterer.labels_
        
#         n_clusters = len(set(labels))
        
#         print(f"threshold: {threshold}, branching_factor: {branching_factor} => Clusters: {n_clusters}")


# # Iterate over a broader range of threshold and branching factor values
# for threshold in [2.0, 2.5, 3.0, 3.5, 4.0]:
#     for branching_factor in [150, 200, 250, 300]:
#         clusterer = Birch(threshold=threshold, branching_factor=branching_factor, n_clusters=None)
#         clusterer.fit(final_df)
#         labels = clusterer.labels_
        
#         n_clusters = len(set(labels))
        
#         print(f"threshold: {threshold}, branching_factor: {branching_factor} => Clusters: {n_clusters}")


# def birch(data_frame):


#     br_hierarchy = Birch(branching_factor = 500, n_clusters = 3, threshold = 1.5)
#     global bir
#     bir=br_hierarchy.fit(data_frame)
#     global b_labels
#     b_labels = br_hierarchy.labels_
#     #h_clust = hierarchy.predict(data_frame)
    
# b_cluster = birch(data_frame=final_df)

# arr_b = np.unique(b_labels)
# #print(arr_b)
# pca_dfb2 = pd.concat([final_df,pd.DataFrame({'cluster':b_labels})],axis=1)

# fig = plt.figure(figsize=(12,8))
# ax = sns.scatterplot(x="pca1",y="pca2",hue='cluster',data=pca_dfb2,palette=['red','green','blue'])#,'yellow','purple','black','pink','orange','maroon'])
# plt.show()


#%%