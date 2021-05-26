#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')


# In[21]:


dataset = pd.read_csv('F:/databases/abalone.csv')


# In[22]:


dataset.shape


# In[23]:


dataset.describe()


# In[25]:


dataset.plot(x='Length', y='Diameter', style='o')  
plt.title('Length vs Diameter')  
plt.xlabel('Length')  
plt.ylabel('Diameter')  
plt.show()


# In[32]:


plt.figure(figsize=(15,10))
plt.tight_layout()
seabornInstance.distplot(dataset['Diameter'])


# In[33]:


X = dataset['Length'].values.reshape(-1,1)
y = dataset['Diameter'].values.reshape(-1,1)


# In[34]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[35]:


regressor = LinearRegression()  
regressor.fit(X_train, y_train) #training the algorithm


# In[36]:


#To retrieve the intercept:
print(regressor.intercept_)

#For retrieving the slope:
print(regressor.coef_)


# In[37]:


y_pred = regressor.predict(X_test)


# In[38]:


df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df


# In[39]:


df1 = df.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# In[43]:


plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()


# In[ ]:




