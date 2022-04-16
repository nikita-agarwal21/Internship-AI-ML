# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 20:54:52 2022

@author: hp
"""
'''
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data=pd.read_csv("mallCustomer.csv")
sns.heatmap(data.isnull())
plt.show()

gender=pd.get_dummies(data['Gender'],drop_first=True)
data.drop(['Gender'],axis=1,inplace=True)
data=pd.concat([data,gender],axis=1)
print(data[:10])
'''

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data=pd.read_csv("mallCustomer.csv")


gen=['Male','Female']
size=data['Gender'].value_counts()
plt.pie(size,labels=gen,autopct="%1.1f%%",explode=(0,0.1),shadow=True)
plt.title('Gender Count')
plt.show()


sns.countplot(data['Age'],palette='hsv',order=[18,22,25,28,30,32,35,38,42,46,50,52,56,60,65,70])
plt.title("Visualization of Age")
plt.show()

ann=data['Annual_Income']
range1=[15,25,35,45,55,65,75,85,95,105,120,130,140]
plt.hist(ann,range1,histtype='bar',rwidth=0.8,color='r')
plt.title('Visualization of Annual income')
plt.xlabel('Annual income')
plt.ylabel('count')
plt.show()


spending=data['Spending_Score ']
sns.countplot(spending,palette='copper',order=[3,6,9,12,15,18,24,27,36,39,42,45,48,51,54,57,60,63,66,69,72,75,78,81,87,90,93,99])
plt.title("Visualization of Spendingscore")
plt.show()





income=data['Annual_Income']
spending=data['Spending_Score ']
x1=income[0:10]
y1=spending[0:10]
plt.subplot(331)
plt.scatter(y1,x1,s=10)
plt.xlabel('income')
plt.ylabel('spending')
plt.title('customer mall')
age=data['Age']
spending=data['Spending_Score ']
x=age[0:10]
y=spending[0:10]
plt.subplot(336)
plt.scatter(y,x,s=10)
plt.xlabel('age')
plt.ylabel('spending')
plt.show()


sns.stripplot(data['Gender'],data['Spending_Score '],palette='Blues',size=5)
plt.title("Gender VS SpendingScore")
plt.show()
 

age_18_25 = data.Age[(data.Age >= 18) & (data.Age <= 25)]
age_26_35 = data.Age[(data.Age >= 26) & (data.Age <= 35)]
age_36_45 = data.Age[(data.Age >= 36) & (data.Age <= 45)]
age_46_55 = data.Age[(data.Age >= 46) & (data.Age <= 55)]
age_55above = data.Age[data.Age >= 56]
agex = ["18-25","26-35","36-45","46-55","55+"]
agey = [len(age_18_25.values),len(age_26_35.values),len(age_36_45.values),len(age_46_55.values),len(age_55above.values)]
plt.figure(figsize=(15,6))
sns.barplot(x=agex, y=agey, palette="mako")
plt.title("Number of Customer and Ages")
plt.xlabel("Age")
plt.ylabel("Number of Customer")
plt.show()


#annual income of genders of different age groups
df=pd.read_csv("mallCustomer.csv")
sns.lmplot(x = "Age", y = "Annual_Income", data = df, hue = "Gender")
plt.title(" Annual income vs spending score vs age")
plt.show()


d=sns.distplot(data['Age'], label='SpendingScore & Age')
d=sns.distplot(data['Spending_Score '])
plt.legend(labels=['Age','SpendingScore '])
d.set(xlabel=None)
plt.show()


x = data['Annual_Income']
y = data['Age']
z = data['Spending_Score ']
sns.lineplot(x, y, color = 'blue')
sns.lineplot(x, z, color = 'pink')
plt.title('Annual Income vs Age and Spending Score', fontsize = 20)
plt.show()

'''
plt.figure(figsize=(10,8), dpi= 80)
sns.pairplot(data, kind="scatter", plot_kws=dict(s=80, edgecolor="white", 
linewidth=2.5))
plt.show()
'''