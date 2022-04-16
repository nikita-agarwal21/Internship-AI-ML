# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 21:23:45 2022

@author: hp
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
data=pd.read_csv("mallCustomer.csv")

#to find k using elbow method
x=data.iloc[:,[3,4]].values
from sklearn.cluster import KMeans
wcs=[]#variance list
for i in range(1,11):
 kmeans=KMeans(n_clusters=i,init='k-means++', max_iter=100,n_init=10,random_state=0) #random points choosed
 kmeans.fit(x)
 wcs.append(kmeans.inertia_)#variance data is added 
plt.plot(range(1,11),wcs,'-o')
plt.xlabel('k value')
plt.ylabel('variance')
plt.title('elbow method ')
plt.show()

kmeans=KMeans(n_clusters=5,init='k-means++',max_iter=100,n_init=10,random_state=0)
y_kmeans=kmeans.fit_predict(x) #which cluster x belonged to is predicted
print(y_kmeans)

plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],s=10,c='green',label='cluster 1')
#x,y cluster is mentioned
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],s=10,c='blue',label='cluster 2')
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],s=10,c='red',label='cluster 3')
plt.scatter(x[y_kmeans==3,0],x[y_kmeans==3,1],s=10,c='yellow',label='cluster 4')
plt.scatter(x[y_kmeans==4,0],x[y_kmeans==4,1],s=10,c='black',label='cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=30,c='cyan',label='centroid')
plt.legend()
plt.title('clusters of customers')
plt.xlabel('annual income')
plt.ylabel('spending score')
plt.show()


#random forest

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
target=pd.DataFrame({'Target':kmeans.labels_})
new_data=pd.concat([data,target],axis=1,sort=False)
#print(new_data.head())
x_new=new_data.drop(['Target'],axis=1)
y_new=new_data['Target']
gen=pd.get_dummies(x_new['Gender'])
x_new=x_new.drop(['Gender'],axis=1)
x_new=pd.concat([x_new,gen],axis=1,sort=False)
x_train,x_test,y_train,y_test=train_test_split(x_new,y_new,test_size=0.20,random_state=2)
dt=DecisionTreeClassifier()
rf=RandomForestClassifier()
model_dt=dt.fit(x_train,y_train)
model_rf=rf.fit(x_train,y_train)
y_pred=model_dt.predict(x_test)
df=pd.DataFrame(y_pred,y_test)
print('decison tree predicted values')
print(df[:6])
print()
y_pred1=model_rf.predict(x_test)
rf1=pd.DataFrame(y_pred1,y_test)
print('random forest predicted values')
print(rf1[:6])


from sklearn import metrics
print("Decision Tree Scores")
print("Accuracy: ",metrics.accuracy_score(y_test,y_pred))
print("MAE (test): ",metrics.mean_absolute_error(y_test, y_pred))
print("MSE (test): ",metrics.mean_squared_error(y_test, y_pred))
print()
print("Random Forest Scores")
print("Accuracy: ",metrics.accuracy_score(y_test,y_pred1))
print("MAE (test): ",metrics.mean_absolute_error(y_test, y_pred1))
print("MSE (test): ",metrics.mean_squared_error(y_test, y_pred1))


pred_data = np.array([[30,10],[70,50],[20,80],[100,80],[100,20],[20,20],[60,60]])
predictions = kmeans.predict(pred_data)
print(predictions)
plt.figure(figsize=(7,7))
plt.scatter(x[:,0], x[:,1], s=20, c=y_kmeans, cmap='gist_rainbow')
plt.scatter(pred_data[:,0], pred_data[:,1], s=250, c=predictions, cmap='gist_rainbow', marker='+')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=75, c='black')
plt.title('Annual income vs spending score distribution')
plt.xlabel('Annual income (k$)')
plt.ylabel('Spending score (1-100)')
plt.show()