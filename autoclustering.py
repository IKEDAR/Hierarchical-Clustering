# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 11:09:07 2021

@author: Kedar
"""
###importing required libraries
import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns

###importing required dataset
df=pd.read_csv(r"D:\Dataset_Assignment Clustering\AutoInsurance.csv")
df.describe
df.columns
df.shape


sns.heatmap(df.isnull())
df.isnull().sum()
df.drop_duplicates()
df.info()

###Taking boxplots to know if there are any outliers
plt.pyplot.boxplot(df['Customer Lifetime Value'])#outliers
plt.pyplot.boxplot(df['Income']) ###No outliers
plt.pyplot.boxplot(df['Monthly Premium Auto'])##outliers
plt.pyplot.boxplot(df['Months Since Last Claim']) ###No outliers
plt.pyplot.boxplot(df['Months Since Policy Inception']) ###No outliers
plt.pyplot.boxplot(df['Number of Open Complaints']) ###outliers
plt.pyplot.boxplot(df['Number of Policies']) ##outliers
plt.pyplot.boxplot(df['Total Claim Amount']) ###outliers

##applying winsorizer to retain outliers
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Customer Lifetime Value','Monthly Premium Auto','Number of Policies','Total Claim Amount'])
df_t = winsor.fit_transform(df[['Customer Lifetime Value','Monthly Premium Auto','Number of Policies','Total Claim Amount']])


df_c=pd.DataFrame(df[['Income','Months Since Last Claim','Months Since Policy Inception']])
df_t=pd.concat([df_t,df_c],axis=1)
df_e=pd.DataFrame(df[['Number of Open Complaints']])


def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)
df_t=norm_func(df_t)
df_t.describe()
df_t.max()
df_t.min()
df_t=pd.concat([df_t,df_e],axis=1)
df_t.min()
df_t.max()

df_n=pd.DataFrame(df[['Response','EmploymentStatus','Gender','Location Code','Marital Status','Policy Type','Renew Offer Type','Vehicle Class']])
from sklearn.preprocessing import OneHotEncoder
# Creating instance of One Hot Encoder
enc = OneHotEncoder() # initializing method

enc_df = pd.DataFrame(enc.fit_transform(df_n).toarray())
enc_df.shape

df_o=pd.DataFrame(df[['Coverage','Education','Vehicle Size']])
from sklearn.preprocessing import LabelEncoder
# creating instance of labelencoder
labelencoder = LabelEncoder()


df_o['Coverage']=labelencoder.fit_transform(df_o['Coverage'])
df_o['Education']=labelencoder.fit_transform(df_o['Education'])
df_o['Vehicle Size']=labelencoder.fit_transform(df_o['Vehicle Size'])

df_final=pd.concat([df_t,enc_df,df_o],axis=1)
###df_final=pd.concat([df_final,df.iloc[:,[0,1]]],axis=1)

from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

z = linkage(df_final, method = "complete", metric = "euclidean")

# Dendrogram
plt.pyplot.figure(figsize=(15, 8));plt.pyplot.title('Hierarchical Clustering Dendrogram');plt.pyplot.xlabel('Index');plt.pyplot.ylabel('Distance')
sch.dendrogram(z, leaf_rotation = 0,  leaf_font_size = 10 )
plt.pyplot.show()


# Now applying AgglomerativeClustering choosing 5 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 7, linkage = 'complete', affinity = "euclidean").fit(df_final) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)

df_final['clust'] = cluster_labels
df_final=pd.concat([df_final,df.iloc[:,[0,1]]],axis=1)
df_final.groupby(df_final.clust).mean()

