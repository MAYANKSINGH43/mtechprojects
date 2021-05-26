#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


# In[8]:


# Import data
training = pd.read_csv('F:\databases\winequality-red.csv')
test = pd.read_csv('F:\databases\winequality-red.csv')


# In[9]:


# Create the X, Y, Training and Test
xtrain = training.drop('quality', axis=1)
ytrain = training.loc[:, 'quality']
xtest = test.drop('quality', axis=1)
ytest = test.loc[:, 'quality']


# In[10]:


# Init the Gaussian Classifier
model = GaussianNB()


# In[11]:


# Train the model
model.fit(xtrain, ytrain)


# In[12]:


# Predict Output
pred = model.predict(xtest)


# In[13]:


# Plot Confusion Matrix
mat = confusion_matrix(pred, ytest)
names = np.unique(pred)
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,
 xticklabels=names, yticklabels=names)
plt.xlabel('Truth')
plt.ylabel('Predicted')


# In[ ]:




