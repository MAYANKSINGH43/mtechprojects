#!/usr/bin/env python
# coding: utf-8

# In[1]:


# from google.colab import drive
# drive.mount("/content/drive")


# In[ ]:


get_ipython().run_line_magic('tensorflow_version', '2.x')
import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))


# change directory to working directory

# In[ ]:


cd /content/drive/My Drive/minor_projectM


# Import library such as numpy, scipy, PIL and os

# In[ ]:


import numpy as np
import pandas as pd
from PIL import Image
import os
import scipy.ndimage


# In[ ]:


import cv2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os


# Define dataset directory to load dataset and the class labels/categories of dataset.

# In[ ]:


DATADIR = "/content/drive/My Drive/minor_projectM"

CATEGORIES = ["original", "1"]


# define size of input image 

# In[ ]:


IMG_SIZE=(1024,1024)
print(IMG_SIZE[0])
print(IMG_SIZE[1])


# Preprocess the data means we have to convert images values into numpy array and resize the image size to size we defined earlier

# In[ ]:


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


# In[ ]:


print(len(data))


# Shuffle the sample data

# In[ ]:


import random

random.shuffle(data)


# In[ ]:


for sample in data[:10]:
    print(sample[1])
    


# initialize the feautre list and label list and save feautre vector X and label vector Y

# In[ ]:


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


# In[ ]:


X.shape


# In[ ]:


len(X)


# In[ ]:


get_ipython().system('pip install tensorflow-GPU')


# In[ ]:


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D


# In[ ]:


plt.imshow(X[220]) # show sample data 


# Defining CNN architecture using model. The sequential shows that model is sequential one.
# Model.add() function is used to add layers (Input, output and Hidden layers)to our model.

# In[ ]:


model = Sequential()

model.add(Conv2D(256, (5, 5), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (5, 5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors


# In[ ]:



model.add(Dense(16))


# In[ ]:



model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[ ]:


history=model.fit(X, y, batch_size=2, epochs=10, validation_split=0.3)


# Model summary is used to summarise the model input and output shae of each layer.

# In[ ]:


model.summary()


# In[ ]:


import keras
from matplotlib import pyplot as plt
#history = model1.fit(train_x, train_y,validation_split = 0.1, epochs=50, batch_size=4)


# In[ ]:


print(history.history.keys())


# plotting train accuracy and validation accuracy per epoch.

# In[ ]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# In[ ]:




