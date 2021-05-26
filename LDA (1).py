#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.model_selection import train_test_split
style.use('fivethirtyeight')
from sklearn.neighbors import KNeighborsClassifier


# In[7]:


# 0. Load in the data and split the descriptive and the target feature
df = pd.read_csv('F:\databases\winequality-red.csv')

X=df.iloc[:,1:].copy()
target=df.iloc[:,0].copy()
df.head()


# In[12]:


X.head()


# In[13]:


target.head()


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X,target,test_size=0.3,random_state=0)


# In[15]:


np.unique(df['fixed acidity'])


# In[17]:


# 2. Compute the mean vector mu and the mean vector per class mu_k
mu = np.mean(X_train,axis=0).values.reshape(13,1) # Mean vector mu --> Since the data has been standardized, the data means are zero 

mu_k = []

for i,Wine in enumerate(np.unique(df['fixed acidity'])):
    mu_k.append(np.mean(X_train.where(df['fixed acidity'] == fixed acidity), axis=0))
mu_k = np.array(mu_k).T


# In[43]:


#covariance matrix
covar_matrix = LDA(n_components = 1)

covar_matrix.fit(X_trans,y)

variance = covar_matrix.explained_variance_ratio_


# In[38]:


#Cumulative sum of variance
var=np.cumsum(np.round(variance, decimals=3)*100)
print("Eigen values\n\n",var)


# In[39]:


#plot for variance explained
plt.ylabel('% Variance Explained')
plt.xlabel('# of Features')
plt.title('LDA Analysis')
plt.ylim(30,100.5)
plt.style.context('seaborn-whitegrid')
plt.plot(var)
plt.show()


# In[40]:


#Fit LDA for two components
lda = LDA(n_components = 1)
LinearComponents = lda.fit_transform(X_trans, y)


# In[44]:


#make it as data frame
finalDf = pd.DataFrame(data = LinearComponents, columns = ['linear discriminant'])

print("After transform X, the linear discriminants are\n\n",finalDf.head())
print("\n")


# In[53]:


#data visualizations
print("2D LDA Visualization\n")

#def visual(df):
#np.random.seed(1)
#sample_size = 9
#df = df.sample(sample_size)
#plt.figure(figsize=(12,9))
#sns.distplot(finalDf['linear discriminant'], hist = True, kde = False, kde_kws = {'linewidth': 3})
#plt.show()
#visual(finalDf)
#print("\n")

def visual1(df):
    np.random.seed(1)
sample_size = 9
plt.figure(figsize=(12,9))
sns.distplot(finalDf['linear discriminant'], hist = True, kde=False, bins=int(180/5), color = 'blue', hist_kws={'edgecolor':'black'})
plt.show()

visual1(finalDf)
print("\n")


# In[59]:


#scatter plot
ax = sns.scatterplot(x="linear discriminant", y="linear discriminant", data=finalDf)
plt.show()
print("\n")

print("The explained variance percentage is:",lda.explained_variance_ratio_*100)


# In[ ]:




