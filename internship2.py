# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 21:21:44 2022

@author: hp
"""


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data=pd.read_csv("mallCustomer.csv")

x=data.iloc[:,:-1]
y=data.iloc[:,4]
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
A=make_column_transformer((OneHotEncoder(categories="auto"),[1]),remainder="passthrough")
x=A.fit_transform(x)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=3)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred=reg.predict(x_test)
df=pd.DataFrame(y_pred,y_test) 
print('Actual vs predicted using linear Regression')
print(df[:10])
col=['Annual_Income']
col1=['Spending_Score ']
 
x=data[col] 
y=data[col1] 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=3)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred=reg.predict(x_test)
#print(y_pred)
plt.style.use('dark_background')
plt.scatter(x_train,y_train,c='yellow')
plt.plot(x_train,reg.predict(x_train),c='r')
plt.title("Actual vs Predicted using linear Regression")
plt.xlabel("Annual_Income")
plt.ylabel("Spending Score")
plt.show()

from sklearn import metrics
print("MSE=",metrics.mean_squared_error(y_test,y_pred))
print("Accuracy=",reg.score(x_test,y_test))

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
own_value=[[60],[50],[140],[144],[148],[150],[155],[160],[164],[168],[170],[175],[178],[180],[184],[188],[192],[195],[198],[250]]
y_pred=reg.predict(own_value)
plt.scatter(x_train,y_train,c='cyan')
plt.plot(own_value,reg.predict(own_value),c='red')
plt.title("Actual vs Predicted using linear Regression")
plt.xlabel("Annual_Income")
plt.ylabel("Spending Score")
plt.show()