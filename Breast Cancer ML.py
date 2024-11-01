#!/usr/bin/env python
# coding: utf-8

# In[35]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from sklearn.cluster import KMeans


# In[3]:


data = pd.read_csv("D:\\02__Programming\\Machine Learning\\DataSets\\Breast Cancer Data.csv")


# In[4]:


data.head()


# In[5]:


data.shape


# In[6]:


data.isnull().sum()


# In[7]:


data.drop(["id","Unnamed: 32"],axis=1,inplace=True)


# In[8]:


data.head()


# In[9]:


data["diagnosis"].value_counts()


# In[10]:


data.iloc[:,1:].describe().iloc[:,:100]


# In[11]:


data.columns


# In[12]:


data.groupby("diagnosis").agg(['sum','mean','max','min'])


# In[13]:


data.iloc[:,1:].corr()


# In[15]:


plt.figure(figsize=(20,20))
sns.heatmap(data.iloc[:,1:].corr(),annot=True,vmin=1,vmax=1)
plt.show()


# In[17]:


x = data.iloc[:,1:].values
y = data.iloc[:,0].values


# In[18]:


sc = StandardScaler()


# In[19]:


x = sc.fit_transform(x)


# In[20]:


x


# In[23]:





# In[29]:


def fun_ML(idx,x,y):
    models = [LogisticRegression(),KNeighborsClassifier(n_neighbors=10),RandomForestClassifier(n_estimators=500),LinearDiscriminantAnalysis(), QuadraticDiscriminantAnalysis()]
    
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
    model = models[idx]
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    return accuracy_score(y_test,y_pred)


# In[30]:


# LogisticRegression
fun_ML(0,x,y)


# In[31]:


# KNeighborsClassifier
fun_ML(1,x,y)


# In[32]:


# RandomForestClassifier
fun_ML(2,x,y)


# In[33]:


# LinearDiscriminantAnalysis
fun_ML(3,x,y)


# In[34]:


# QuadraticDiscriminantAnalysis
fun_ML(4,x,y)


# In[36]:


# Un supervise learning 
kmc = KMeans(n_clusters=2)
kmc.fit(x)


# In[37]:


kmc.labels_


# In[38]:


y


# In[40]:


kmc.labels_.sum()


# In[ ]:




