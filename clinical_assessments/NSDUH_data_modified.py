#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 16:33:34 2022

@author: kaloso
"""

# %% IMPORTING LIBRARIES
import sys
import random 
from numpy import unique
from numpy import where
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from IPython.display import display
from sklearn.cluster import KMeans 
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import homogeneity_score, completeness_score, \
v_measure_score, adjusted_rand_score, adjusted_mutual_info_score, silhouette_score
import seaborn as sns
from sklearn import preprocessing
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from matplotlib import style
from numpy import asarray
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances_argmin
from scipy.stats import chi2_contingency
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix,zero_one_loss
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn import utils
import warnings
from sklearn.metrics import confusion_matrix,classification_report
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc
from sklearn.preprocessing import normalize
from sklearn.cluster import OPTICS, cluster_optics_dbscan
import matplotlib.gridspec as gridspec
float_formatter = lambda x: "%.3f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import pairwise_distances
from matplotlib import pyplot as plt
import networkx as nx
from pandas.plotting import scatter_matrix
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn import preprocessing, decomposition
from matplotlib.collections import LineCollection
import time
import operator
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import dendrogram
import math
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.cluster import Birch
from sklearn.cluster import kmeans_plusplus
import hdbscan

#from functions import *
sns.set()
warnings.filterwarnings('ignore')

# %% [1] IMPORTING DATA

alcohol = pd.read_excel("Documents/MEng_Research/NSDUH/2019_Data/DATA-Alcohol.xlsx")
marijuana = pd.read_excel("Documents/MEng_Research/NSDUH/2019_Data/DATA-Marijuana.xlsx")
cocaine = pd.read_excel("Documents/MEng_Research/NSDUH/2019_Data/DATA-Cocaine.xlsx")
#sns.heatmap(marijuana.isnull(),yticklabels=False, cbar=False,cmap='Blues')

#print(alcohol.isnull().sum())

#print(df.head())


alcohol.loc[(alcohol['GRSKBNGDLY'].isnull()==True),'GRSKBNGDLY'] = alcohol['GRSKBNGDLY'].mean()
alcohol.loc[(alcohol['GRSKBNGWK'].isnull()==True),'GRSKBNGWK'] = alcohol['GRSKBNGWK'].mean()
marijuana.loc[(marijuana['GRSKMRJMON'].isnull()==True),'GRSKMRJMON'] = marijuana['GRSKMRJMON'].mean()
marijuana.loc[(marijuana['GRSKMRJWK'].isnull()==True),'GRSKMRJWK'] = marijuana['GRSKMRJWK'].mean()
marijuana.loc[(marijuana['DIFOBTMRJ'].isnull()==True),'DIFOBTMRJ'] = marijuana['DIFOBTMRJ'].mean()
cocaine.loc[(cocaine['GRSKCOCMON'].isnull()==True),'GRSKCOCMON'] = cocaine['GRSKCOCMON'].mean()
cocaine.loc[(cocaine['GRSKCOCWK'].isnull()==True),'GRSKCOCWK'] = cocaine['GRSKCOCWK'].mean()



#print(marijuana.isnull().sum())
#print(cocaine.isnull().sum())
#print(alcohol.duplicated().sum())

df = alcohol.iloc[0:25000,:]

df_val = alcohol.iloc[25000:50000,:]


#print(df.describe())

# %% [2] PREPROCESSING DATA

data_scale = normalize(df)
df_scaled = pd.DataFrame(data_scale, columns=df.columns)

# data_scale_val = normalize(df_val)
# df_scaled_val = pd.DataFrame(data_scale_val, columns=df_val.columns)

scaler = StandardScaler()
data = scaler.fit_transform(df)

scaler_val = StandardScaler()
data_val = scaler.fit_transform(df_val)


#HEATMAP

# corr = df_scaled.corr(method='spearman')
# f,ax = plt.subplots(figsize=(12,9))
# cmap = sns.diverging_palette(10, 275,as_cmap=True )
# sns.heatmap(corr, cmap=cmap,square=True,linewidths=1,cbar_kws={'shrink':0.5},ax=ax)


# %% [3] PROCESSING DATA

pca = PCA(n_components=2,random_state=123)
principal_comp = pca.fit_transform(data)
pca_df = pd.DataFrame(data=principal_comp,columns=['pca1','pca2'])

pca_val = PCA(n_components=2,random_state=123)
principal_comp_val = pca.fit_transform(data_val)
pca_df_val = pd.DataFrame(data=principal_comp_val,columns=['pca1_val','pca2_val'])


# %% [4] K-MEANS CLUSTERING

#ELBOW METHOD

ks = range(1, 10)
inertias = []

for k in ks:
    model = KMeans(n_clusters=k)
    model.fit(data)
    inertias.append(model.inertia_)

# plt.figure(figsize=(8,5))
# plt.style.use('bmh')
# plt.plot(ks, inertias, '-o')
# plt.xlabel('Number of clusters, k')
# plt.ylabel('Inertia')
# plt.xticks(ks)
# plt.show()



#print(pca.explained_variance_ratio_)

def K_means(data_frame):

    k_means = KMeans(n_clusters = 2, random_state=123, n_init=30)
    global me
    me=k_means.fit(data_frame)
    global c_labels
    c_labels = k_means.labels_
    
    
k_cluster=K_means(data_frame=pca_df)

arr_k = np.unique(c_labels)
#print(arr_k) 
#print(c_labels[:1000])
pca_dfk2 = pd.concat([pca_df,pd.DataFrame({'cluster':c_labels})],axis=1)

# fig = plt.figure(figsize=(12,8))
# ax = sns.scatterplot(x="pca1",y="pca2",hue='cluster',data=pca_dfk2,palette=['red','green'])#,'blue','yellow','purple','black','pink','orange','maroon'])
# plt.show()

# %%  [5] K-MEANS ++ INITIALIZATION

def K_means_plus(data_frame):

    k_means = KMeans(n_clusters = 2, random_state=None, n_init=10,init='k-means++',algorithm='auto')
    global plus
    plus=k_means.fit(data_frame)
    global m_labels
    m_labels = k_means.labels_
    
    
km_cluster=K_means_plus(data_frame=pca_df)

arr_km = np.unique(m_labels)
#print(arr_km) 
#print(c_labels[:1000])
pca_dfkm = pd.concat([pca_df,pd.DataFrame({'cluster':m_labels})],axis=1)

# fig = plt.figure(figsize=(12,8))
# ax = sns.scatterplot(x="pca1",y="pca2",hue='cluster',data=pca_dfkm,palette=['red','green'])#,'blue','yellow','purple','black','pink','orange','maroon'])
# plt.show()



# %% [6] HIERARCHICAL CLUSTERING


def Agglomerative(n_clust, data_frame):


    hierarchy = AgglomerativeClustering(n_clusters = n_clust, affinity='euclidean', linkage='ward',compute_full_tree=True)
    global hie
    hie=hierarchy.fit(data_frame)
    global h_labels
    h_labels = hierarchy.labels_
    #h_clust = hierarchy.predict(data_frame)
    
h_cluster = Agglomerative(n_clust=5, data_frame=pca_df)
arr_h = np.unique(h_labels)
#print(arr_h)
pca_dfh2 = pd.concat([pca_df,pd.DataFrame({'cluster':h_labels})],axis=1)

# fig = plt.figure(figsize=(12,8))
# ax = sns.scatterplot(x="pca1",y="pca2",hue='cluster',data=pca_dfh2,palette=['red','green','blue','yellow','purple'])#,'black','pink','orange','maroon'])
# plt.show()



# %% [7] BIRCH

def birch(data_frame):


    br_hierarchy = Birch(branching_factor = 50, n_clusters = None, threshold = 1.5)
    global bir
    bir=br_hierarchy.fit(data_frame)
    global b_labels
    b_labels = br_hierarchy.labels_
    #h_clust = hierarchy.predict(data_frame)
    
b_cluster = birch(data_frame=pca_df)

arr_b = np.unique(b_labels)
#print(arr_b)
pca_dfb2 = pd.concat([pca_df,pd.DataFrame({'cluster':b_labels})],axis=1)

# fig = plt.figure(figsize=(12,8))
# ax = sns.scatterplot(x="pca1",y="pca2",hue='cluster',data=pca_dfh2,palette=['red','green','blue','yellow','purple'])#,'black','pink','orange','maroon'])
# plt.show()



# %% [8] DBSCAN


def Dbscan(data_frame):

    dbscan = DBSCAN(eps=0.5,min_samples=5,metric='euclidean')
    global db
    db=dbscan.fit(data_frame)
    global d_labels
    d_labels = dbscan.labels_
    #d_clust = dbscan.predict(data_frame)
    
d_cluster = Dbscan(data_frame=pca_df)
arr_d = np.unique(d_labels)
#print(arr_d)

#print (d_labels[:1000])


pca_dfd2 = pd.concat([pca_df,pd.DataFrame({'cluster':d_labels})],axis=1)
# fig = plt.figure(figsize=(12,8))
# ax = sns.scatterplot(x="pca1",y="pca2",hue='cluster',data=pca_dfd2,palette=['red','green','blue','yellow'])#,'purple','black','pink','orange','maroon'])
# plt.show()


#%% HDBSCAN

clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
    gen_min_span_tree=False,
    metric='euclidean', min_cluster_size=50, min_samples=5000, p=None)

clusterer.fit(pca_df)

#print(np.unique(clusterer.labels_))


hdbscan = hdbscan.HDBSCAN(min_cluster_size=10, min_samples = 5)
labels = hdbscan.fit_predict(data)
#hdbscan.condensed_tree_.plot(select_clusters=True)


hd_labels = clusterer.labels_
print (np.unique(hd_labels))

pca_dfhd = pd.concat([pca_df,pd.DataFrame({'cluster':hd_labels})],axis=1)
# fig = plt.figure(figsize=(12,8))
# ax = sns.scatterplot(x="pca1",y="pca2",hue='cluster',data=pca_dfd2,palette=['red','green','blue','yellow'])#,'purple','black','pink','orange','maroon'])
# plt.show()


 
# %% [10] IMPLEMENTING CROSS VALIDATION
X = pca_dfhd.iloc[:,:-1]
y = pca_dfhd.iloc[:,-1]


  
# %% [11] SILHOUTTE SCORE
score = silhouette_score(X,y,metric='euclidean',sample_size=None, random_state=None)
print('Silhoutte:%3f' %score)


# %% [12] K-FOLD CROSS VALIDATION

k = 10
kf = KFold(n_splits=k, random_state=None)
model2 = LogisticRegression(multi_class='multinomial',solver= 'lbfgs')

acc_score = []
 
for train_index , test_index in kf.split(X):
    #model_train, model_test, cluster_train, cluster_test = data[train_index],data[test_index],c_labels[train_index],c_labels[test_index]
    X_train , X_test = X.iloc[train_index,:],X.iloc[test_index,:]
    y_train , y_test = y[train_index] , y[test_index]
     
    model2.fit(X_train,y_train)
    pred_values = model2.predict(X_test)
     
    acc = accuracy_score(pred_values , y_test)
    acc_score.append(acc)
     
avg_acc_score = sum(acc_score)/k
 
print('accuracy of each fold - {}'.format(acc_score))
print('Avg accuracy : {}'.format(avg_acc_score))

# %% [13] ERROR DEVIATION

mape = mean_absolute_error(y_test,pred_values)*100
#print(mape)

rmse = mean_squared_error(y_test,pred_values)
print(math.sqrt(rmse))


#%%





































