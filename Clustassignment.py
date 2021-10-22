# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 18:14:45 2021

@author: Kedar
"""
##importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
import feature_engine

##importing dataset and converting into a dataframe
df1=pd.read_excel(r"D:\Dataset_Assignment Clustering\EastWestAirlines.xlsx",sheet_name='data')
##getting info about dataset
df1.columns
df1.shape
df1.var()
df1.describe()
##dropping duplicated columns
df1.drop_duplicates(inplace=True)
df1.var()
df1.dtypes

##typecasting dataframe to 'float64' 
df1=df1.astype('float64')


###Getting boxplots to know if there are any outliers
plt.pyplot.boxplot(df1['Balance'])
plt.pyplot.boxplot(df1['Qual_miles'])
plt.pyplot.boxplot(df1['Bonus_miles'])
plt.pyplot.boxplot(df1['Bonus_trans'])
plt.pyplot.boxplot(df1['Flight_miles_12mo'])
plt.pyplot.boxplot(df1['Flight_trans_12'])
plt.pyplot.boxplot(df1['Days_since_enroll'])

###removing columns who have no outliers so that winsorization can be performed
df2=df1[['cc1_miles','cc2_miles','cc3_miles','Days_since_enroll','Qual_miles']]
df2.shape
df1=df1.drop(['cc1_miles','cc2_miles','cc3_miles','Days_since_enroll','Qual_miles','ID#'],axis=1)
              

###winsorization
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Balance','Bonus_miles','Bonus_trans','Flight_miles_12mo','Flight_trans_12'])
df1 = winsor.fit_transform(df1[['Balance','Bonus_miles','Bonus_trans','Flight_miles_12mo','Flight_trans_12']])

##Checking if there are any outliers
plt.pyplot.boxplot(df1['Balance'])
plt.pyplot.boxplot(df1['Bonus_miles'])
plt.pyplot.boxplot(df1['Bonus_trans'])
plt.pyplot.boxplot(df1['Flight_miles_12mo'])
plt.pyplot.boxplot(df1['Flight_trans_12'])

##getting the entire dataset in one dataframe
df3=pd.concat([df1,df2],axis=1)             
df1=df3.iloc[:,[9,0,2,3,4,5]]

##performing normalization function
def norm_func(i):
	x = (i-i.min())	/(i.max()-i.min())
	return(x)
df1=norm_func(df1)
df1.var

df1=pd.concat([df1,df3.iloc[:,[1,6,7,8]]],axis=1)


##performing clustering and getting dendrogram
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

z = linkage(df1, method = "complete", metric = "euclidean")

#Dendrogram
plt.pyplot.figure(figsize=(15,8));plt.title=('Hierarchical clustering dendrogram');plt.xlable=('Index');plt.ylabel=('Distance')
sch.dendrogram(z,leaf_rotation=0,leaf_font_size=10)
plt.pyplot.show()

###performing agglomerative clustering
from sklearn.cluster import AgglomerativeClustering
h_complete = AgglomerativeClustering(n_clusters = 3, linkage = 'complete', affinity = "euclidean").fit(df1) 
h_complete.labels_

###Adding clusters column to dataframe
cluster_labels = pd.Series(h_complete.labels_)
df1['clust']=cluster_labels

###rearranging dataframes
df1=df1.iloc[:,[10,0,1,2,3,4,5,6,7,8,9]]
df1.head()
# Aggregate mean of each cluster
df1.iloc[:,1:].groupby(df1.clust).mean()

#creating a new excel file
df1.to_excel('Eastwestclusters.xlsx',encoding='utf8')

