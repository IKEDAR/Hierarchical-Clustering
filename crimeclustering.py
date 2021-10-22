# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 17:24:06 2021

@author: Kedar
"""
###importing all required libraries
import pandas as pd
import matplotlib as plt
import numpy as np
import seaborn as sns

##importing dataset
df=pd.read_csv('D:\Dataset_Assignment Clustering\crime_data.csv')
df.describe()
df.shape
df.var()

plt.pyplot.boxplot(df['Murder'])
plt.pyplot.boxplot(df['Assault'])
plt.pyplot.boxplot(df['UrbanPop'])
plt.pyplot.boxplot(df['Rape'])

plt.pyplot.hist(df['Murder'])
plt.pyplot.hist(df['Assault'])
plt.pyplot.hist(df['UrbanPop'])
plt.pyplot.hist(df['Rape'])

df.mean()
df.median()
df.mode()
df.skew()
df.kurt()

df1=pd.DataFrame(df['Unnamed: 0'])
df.drop(['Unnamed: 0'],axis=1,inplace=True)
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)
df = norm_func(df)
df.describe()



# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

z = linkage(df, method = "complete", metric = "euclidean")

# Dendrogram
plt.pyplot.figure(figsize=(15, 8));plt.pyplot.title('Hierarchical Clustering Dendrogram');plt.pyplot.xlabel('Index');plt.pyplot.ylabel('Distance')
sch.dendrogram(z, leaf_rotation = 0,  leaf_font_size = 10 )
plt.pyplot.show()


# Now applying AgglomerativeClustering choosing 5 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 4, linkage = 'complete', affinity = "euclidean").fit(df) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)

df['clust'] = cluster_labels # creating a new column and assigning it to new column 
df1=pd.DataFrame(df1)
df=pd.concat([df,df1],axis=1)
df=df.iloc[:,[4,5,0,1,2,3]]
df=df.rename(columns={'Unnamed: 0':'States','0':'Murder','1':'Assault','2':'Urban Pop','3':'Rape'},inplace=True)



# Aggregate mean of each cluster
df.iloc[:, 2:].groupby(df.clust).mean()
