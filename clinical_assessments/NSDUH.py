import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from matplotlib import style
import warnings
from numpy import asarray
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings('ignore')

#dataset = pd.read_csv("Documents/MEng Research/NSDUH/NSDUH_2018_Tab.tsv")

df = pd.read_csv("C:/Users/Kaloso/Documents/MEng Research/A Survey on Addiction (Responses).csv")
df.drop('Timestamp',axis=1,inplace=True)
df.drop('Sex',axis=1,inplace=True)
#print(df.head())

#print(df.iloc[:,:])

df['Q1'].replace (['Everyday','Once week','Twice a week','Twice a day'],[3,1,2,4],inplace=True)
df['Q2'].replace (['Yes','No','Not really'],[2,1,1.5],inplace=True)
df['Q3'].replace (['Yes','No','Not sure'],[2,1,1.5],inplace=True)
df['Q4'].replace (['Does not matter','More than three','Two to three','Alone'],[1,2,3,4],inplace=True)
df['Q5'].replace (['Many','10 more','Five more','Three more'],[1,1.5,2,2.5],inplace=True)
df['Q6'].replace (['Many','Around 10','Three to Five','One or two'],[1,1.5,2,2.5],inplace=True)
df['Q7'].replace (['Yes','No','Not really'],[2,1,1.5],inplace=True)
df['Q8'].replace (['Yes','No','Manageable'],[2,1,1.5],inplace=True)
df['Q9'].replace (['No, I keep it to myself','Close friends and family only','Selected few','Everybody knows about it'],[2.5,2,1.5,1],inplace=True)
df['AD'].replace (['Manageable','Addicted','Not addicted',"I don't know"],[2,3,0,1],inplace=True)

# corr = df.corr(method='spearman')
# f,ax = plt.subplots(figsize=(12,9))
# cmap = sns.diverging_palette(10, 275,as_cmap=True )

# sns.heatmap(corr, cmap=cmap,square=True,linewidths=1,cbar_kws={'shrink':0.5},ax=ax)  


x = df.iloc[:,0:-1].values
y = df.iloc[:,-1].values
print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 4)

sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

model_SVC = SVC(kernel = 'rbf', random_state = 4)
model_SVC.fit(x_train, y_train)

y_pred_svm = model_SVC.decision_function(x_test)

model_logistic = LogisticRegression()
model_logistic.fit(x_train, y_train)

y_pred_logistic = model_logistic.decision_function(x_test)

model_ANN = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=4)
model_ANN.fit(x_train,y_train)

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(x, y)


print(model_logistic.score(x_test,y_test))
print(model_SVC.score(x_test,y_test))
print(model_ANN.score(x_test,y_test))
print(neigh.score(x_test,y_test))

# logistic_fpr, logistic_tpr, threshold = roc_curve(y_test, y_pred_logistic)
# auc_logistic = auc(logistic_fpr, logistic_tpr)

# svm_fpr, svm_tpr, threshold = roc_curve(y_test, y_pred_svm,pos_label=0)
# auc_svm = auc(svm_fpr, svm_tpr)

# plt.figure(figsize=(5, 5), dpi=100)
# plt.plot(svm_fpr, svm_tpr, linestyle='-', label='SVM (auc = %0.3f)' % auc_svm)
# plt.plot(logistic_fpr, logistic_tpr, marker='.', label='Logistic (auc = %0.3f)' % auc_logistic)

# plt.xlabel('False Positive Rate -->')
# plt.ylabel('True Positive Rate -->')

# plt.legend()

# plt.show()

#UNIVARIATE SELECTION

# bestfeatures = SelectKBest(score_func=chi2, k=9)

# fit = bestfeatures.fit(df_norm,y)

# dfscores = pd.DataFrame(fit.scores_)

# dfcolumns = pd.DataFrame(df_norm.columns) # #concat two dataframes for better visualization
                                     
# featureScores = pd.concat([dfcolumns,dfscores],axis=1)

# featureScores.columns = ['Specs','Score']  #naming the dataframe columns

# print(featureScores.nlargest(9,'Score'))  #print 10 best features

#FEATURE IMPORTANCE

# model = ExtraTreesClassifier()
# model.fit(df_norm,y)
# feat_importances = pd.Series(model.feature_importances_, index=df_norm.columns)
# feat_importances.nlargest(10).plot(kind='barh')
# plt.show()  #print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers

#FEATURE CORRELATION 

# corrmat = df.corr()
# top_corr_features = corrmat.index
# plt.figure(figsize=(15,15))
# g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")  #plot heat map

#FEATURE IMPORTANCES WITH FOREST TREES























# marijuana = pd.read_excel("Documents/MEng Research/NSDUH/DATA-Marijuana.xlsx")
# #print(df2.head())

# cocaine = pd.read_excel("Documents/MEng Research/NSDUH/DATA-Cocaine.xlsx")
# #print(df3.head())

# stimulants = pd.read_excel("Documents/MEng Research/NSDUH/DATA-Stimulants.xlsx")
# #print(df4.head())


#sns.swarmplot(x='ALCWD2SX',y='QUESTID2',data=df)
