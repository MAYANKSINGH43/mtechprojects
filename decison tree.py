#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import sklearn
import pydotplus


# In[2]:


my_data=pd.read_csv(r"F:\databases\pulsar_stars.csv")
print(my_data)


# In[3]:


train_set=my_data.iloc[:14319,:9]
print(train_set)


# In[4]:


test_set=my_data.iloc[14319:17898,:9]
print(test_set)


# In[5]:


train_set_x=train_set.iloc[:,: -1]
train_set_y=train_set.iloc[:,-1]

test_set_x=test_set.iloc[:,: -1]
test_set_y=test_set.iloc[:,-1]


# In[6]:


from sklearn.tree import DecisionTreeClassifier
star_tree=DecisionTreeClassifier(criterion="entropy",max_depth=10)

print(star_tree)
star_tree.fit(train_set_x,train_set_y)

predict_tree=star_tree.predict(test_set_x)
print(predict_tree[0:5]) # predicted output labels of test dataset 
print(test_set_y[0:5])   # actual label of test dataset


# In[7]:


from sklearn import *
from sklearn.metrics import *
print("\ndecision tree accuracy:",metrics.accuracy_score(test_set_y,predict_tree))


# In[8]:


get_ipython().system('pip install six')


# In[9]:


from sklearn.externals.six import StringIO
import pydotplus
from graphviz import *

featureNames = my_data.columns[0:8]
targetNames = my_data['target_class'].unique().tolist()
dot_data = StringIO()  #Text I/O implementation using an in-memory buffer.

#tree.export_graphviz(decision_tree, out_file=None, max_depth=None,
#                   feature_names=None, class_names=None, label='all',
#                   filled=False, leaves_parallel=False, impurity=True,
#                  node_ids=False, proportion=False, rotate=False,
#                 rounded=False, special_characters=False, precision=3):
#This function generates a GraphViz representation of the decision tree,
#which is then written into `out_file`

dot_data = tree.export_graphviz(star_tree, out_file=None ,
                                feature_names=featureNames,
                                filled=True, rotate=False)
#Load graph as defined by data in DOT format.
#The data is assumed to be in DOT format. 
#It will be parsed and a Dot class will be returned, representing the graph.
graph=pydotplus.graph_from_dot_data(dot_data)


# In[10]:


from graphviz import *


# In[11]:


from sklearn import tree


# In[12]:


tree.plot_tree(star_tree)


# In[15]:


import matplotlib.pyplot as plt
fn=['Mean_of_the_integrated_profile','Standard_deviation_of_the_integrated_profile','Excess_kurtosis_of_the_integrated_profile','Skewness_of_the_integrated_profile','Mean_of_the_DM-SNR_curve','Standard_deviation_of_the_DM-SNR_curve','Excess_kurtosis_of_the_DM-SNR_curve','Skewness_of_the_DM-SNR_curve']
cn=['0','1']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (3,3), dpi=1000)
tree.plot_tree(star_tree,
               feature_names = fn, 
               class_names=cn,
               filled = True);
fig.savefig('imagename.png')


# In[ ]:




