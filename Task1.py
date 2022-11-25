#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import numpy as np


# In[ ]:


random.seed(42)


# In[30]:


initializers = ["glorot_uniform", 
                keras.initializers.RandomNormal(mean=0.0, stddev=0.05), 
                keras.initializers.RandomUniform(minval=-0.05, maxval=0.05), 
                keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05),
                keras.initializers.Zeros()]
initializer = initializers[0]

activations = ["relu", "softmax", "sigmoid", "tanh", "linear"]
activation = activations[0]

optimizers = ["sgd", "rmsprop", "adam", "adadelta"]
optimizer = optimizers[0]

regularizers = [None, keras.regularizers.L1(l1=0.01), keras.regularizers.L2(l2=0.01)]
regularizer = regularizers[0]


# In[31]:


fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
X_train_full, X_test = X_train_full/255, X_test/255
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
plt.imshow(X_train_full[0])
print(y_train_full)


# In[32]:


print(tf.random.uniform([1])) 


# In[33]:


model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation=activation, kernel_initializer=initializer, kernel_regularizer=regularizer))
model.add(keras.layers.Dense(100, activation=activation, kernel_initializer=initializer, kernel_regularizer=regularizer))
model.add(keras.layers.Dense(10, activation="softmax", kernel_initializer=initializer, kernel_regularizer=regularizer))
model.summary()


# In[34]:


model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"]
)


# In[35]:


history = model.fit(X_train_full, y_train_full, epochs=4, batch_size=128)


# In[36]:


model.evaluate(X_test, y_test)


# In[37]:


def doedingen(Activation, Initializer, Regularizer, Optimizer):
  random.seed(42)
  model = keras.models.Sequential()
  model.add(keras.layers.Flatten(input_shape=[28, 28]))
  model.add(keras.layers.Dense(300, activation=Activation, kernel_initializer=Initializer, kernel_regularizer=Regularizer))
  model.add(keras.layers.Dense(100, activation=Activation, kernel_initializer=Initializer, kernel_regularizer=Regularizer))
  model.add(keras.layers.Dense(10, activation="softmax", kernel_initializer=Initializer, kernel_regularizer=Regularizer))
  model.summary()
  model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=Optimizer,
    metrics=["accuracy"]
  )
  history = model.fit(X_train_full, y_train_full, epochs=4, batch_size=128)
  model.evaluate(X_test, y_test)
  return round(model.evaluate(X_test, y_test)[1], 4) 



# Making a dictionary with the accuracy results of different parameters:

# In[ ]:


answers = {}

params = [initializers, activations, optimizers, regularizers]
paramnames = ["initializers", "activations", "optimizers", "regularizers"]
for i in range(len(params)):
  answers[paramnames[i]] = []
  Initializer = params[0][0]
  Activation = params[1][0]
  Optimizer = params[2][0]
  Regularizer = params[3][0]
  PARA = [Initializer, Activation, Optimizer, Regularizer]
  for j in range(len(params[i])):
    PARA[i] = params[i][j]
    ans = doedingen(PARA[1], PARA[0], PARA[3], PARA[2])
    answers[paramnames[i]].append(str(params[i][j])+" "+str(ans))


# In[40]:


for i in answers:
    print(i)
    for j in answers[i]:
      print("\t", j)


# In[ ]:




