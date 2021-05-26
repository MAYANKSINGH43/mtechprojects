#!/usr/bin/env python
# coding: utf-8

# In[1]:


from google.colab import drive
drive.mount("/content/drive")


# In[2]:


get_ipython().run_line_magic('tensorflow_version', '2.x')
import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))


# In[3]:


cd /content/drive/My Drive/Colab Notebooks/Python Practice/minor_projectM


# In[4]:


import numpy as np
import pandas as pd
from PIL import Image
import os
import scipy.ndimage


# In[5]:


IMG_SIZE=(256,256)
print(IMG_SIZE[0])
print(IMG_SIZE[1])


# In[6]:


import cv2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os

DATADIR = "/content/drive/My Drive/Colab Notebooks/Python Practice/minor_projectM"

CATEGORIES = ["original", "1"]



# In[7]:


data = []


def create_data():
    for category in CATEGORIES:  

        path = os.path.join(DATADIR,category) 
        class_num = CATEGORIES.index(category)  
        for img in tqdm(os.listdir(path)):  
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_COLOR)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE[0], IMG_SIZE[1]))  # resize to normalize data size
                data.append(np.array([new_array, class_num])) # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
            #except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            #except Exception as e:
            #    print("general exception", e, os.path.join(path,img))
 
create_data()


# In[8]:


print(len(data))


# In[9]:


import random

random.shuffle(data)


# In[10]:


for sample in data[:10]:
    print(sample[1])
    


# In[11]:


X = []
y = []

for features,label in data:
    X.append(features)
    y.append(label)

print(X[1].reshape(-1,IMG_SIZE[0], IMG_SIZE[1], 3))

X = np.array(X).reshape(-1,IMG_SIZE[0], IMG_SIZE[1], 3)
X=X/255;
X.size
y=np.array(y)


# In[12]:


X.shape


# In[13]:


len(X)


# In[14]:


get_ipython().system('pip install tensorflow-GPU')


# In[15]:


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D


# In[16]:


plt.imshow(X[220])


# In[17]:


model = Sequential()

model.add(Conv2D(256, (5, 5), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (5, 5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors


# In[18]:



model.add(Dense(16))


# In[19]:



model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[20]:


history = model.fit(X, y, batch_size=2, epochs=10, validation_split=0.3)


# In[21]:


model.summary()

