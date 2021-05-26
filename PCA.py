#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from IPython import display
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[43]:


#Reading the dataset
dataset = pd.read_csv(r"F:\databases\Website Phishing.csv")


# In[44]:


dataset.info()


# In[45]:


dataset.describe()


# In[46]:


dataset.head()


# In[47]:


p=sns.pairplot(dataset, hue = 'Result')


# In[48]:


plt.figure(figsize=(15,15))
p=sns.heatmap(dataset.corr(), annot=True,cmap='RdYlGn')


# In[49]:


dataset.hist(figsize=(15,12),bins = 15)
plt.title("Features Distribution")
plt.show()


# In[50]:


X =dataset.drop(['Result'],axis=1)
y = dataset.Result


# In[51]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X=scaler.fit_transform(X)


# In[52]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=20, stratify=y)


# In[53]:


knn = KNeighborsClassifier(7)
knn.fit(X_train,y_train)
print("Train score before PCA",knn.score(X_train,y_train),"%")
print("Test score before PCA",knn.score(X_test,y_test),"%")


# In[54]:


from sklearn.decomposition import PCA
pca = PCA()
X_new = pca.fit_transform(X)


# In[55]:


pca.get_covariance()


# In[56]:


explained_variance=pca.explained_variance_ratio_
explained_variance


# In[57]:


with plt.style.context('dark_background'):
    plt.figure(figsize=(6, 4))
    plt.bar(range(9), explained_variance, alpha=0.5, align='center', label='individual explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()


# In[58]:


pca=PCA(n_components=3)
X_new=pca.fit_transform(X)


# In[59]:


pca=PCA(n_components=3)
X_new=pca.fit_transform(X)


# In[60]:


X_train_new, X_test_new, y_train, y_test = train_test_split(X_new, y, test_size = 0.3, random_state=20, stratify=y)


# In[61]:


knn_pca = KNeighborsClassifier(7)
knn_pca.fit(X_train_new,y_train)
print("Train score after PCA",knn_pca.score(X_train_new,y_train),"%")
print("Test score after PCA",knn_pca.score(X_test_new,y_test),"%")


# In[62]:


classifier = knn_pca
from matplotlib.colors import ListedColormap
X_set, y_set = X_test_new, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel(),np.zeros((X1.shape[0],X1.shape[1])).ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('pink', 'lightgreen','lightyellow')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green','blue'))(i), label = j)
plt.title('KNN PCA (Test set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()


# In[ ]:




