#
# Implementation of Relation Networks using MNIST.
# https://arxiv.org/pdf/1706.01427.pdf
#
from __future__ import print_function
import keras
import numpy as np
from keras.datasets import mnist
from keras import backend as K
from keras.layers import Input, Dense, Reshape, Lambda, Add, Conv2D, TimeDistributed, MaxPooling2D
from keras.models import Model
from random import *

#
# Environment Parameters
#
batch_size = 128
num_classes = 10
epochs = 12
img_rows, img_cols = 28, 28  # input image dimensions

# Load & Preprocess MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = keras.utils.to_categorical(y_train, num_classes) # convert class vectors to binary class matrices
y_test = keras.utils.to_categorical(y_test, num_classes)

#
# Returns relation vectors from an input convolution tensor map.
# A relation vector is the concatenation of two objects, 
#     in this case the objects are "pixels" of the tensor.
#
def getRelationVectors(x):
	objects = []
	relations = []
	shape = K.int_shape(x)
	k = 25     # Hyperparameter which controls how many objects are considered
	width = 2  # Width of tensor "pixel"

	# Get k random objects
	for time in range(k):
		i = randint(width, shape[1] - 1 - width)
		j = randint(width, shape[2] - 1 - width)
		objects.append(x[:, i - width:i + width, j - width:j + width, :])

	# Concatenate each pair of objects to form a relation vector
	for i in range(len(objects)):
		for j in range(i, len(objects)):
			relations.append(K.concatenate([objects[i], objects[j]], axis=1))

	# Restack objects into Keras tensor [batch, relation_ID, relation_vectors]
	return K.permute_dimensions(K.stack([r for r in relations], axis=0), [1, 0, 2])

#
# Define CNN
#
inputs = Input(shape=input_shape)
x = Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)
x = Conv2D(32, (3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
shape = K.int_shape(x)

#
# Define Relation Network layer
#
RN_inputs = Input(shape=(1, 2 * shape[3]))
RN_x = Dense(64, activation='relu')(RN_inputs)
RN_outputs = Dense(32, activation='relu')(RN_x)
RN = Model(inputs=RN_inputs, outputs=RN_outputs)

#
# Implements g_theta.
# Use TimeDistributed layer to apply the same RN module
#     on all rows of the relation list.
#
relations = Lambda(getRelationVectors)(x)
g = TimeDistributed(RN)(relations)
g = Lambda(lambda x: K.sum(x, axis=1))(g)

f = Dense(32, activation='relu')(g)
outputs = Dense(10, activation='softmax')(f)

#
# Train model
#
model = Model(inputs=inputs, outputs=outputs)
print(model.summary())
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)