#
# Implementation of Relation Networks using CLEVR.
# https://arxiv.org/pdf/1706.01427.pdf
#
from __future__ import print_function
import keras
import numpy as np
from keras import backend as K
from keras.layers import Input, Dense, Reshape, Lambda, Conv2D, TimeDistributed, MaxPooling2D
from keras.models import Model
from random import *

#
# Retrieves and processes CLEVR dataset
#
def load_data():
	datapath = ""

#
# Environment Parameters
#
batch_size = 64
num_classes = 10
epochs = 10

# Load & Preprocess CLEVR
(x_train, y_train), (x_test, y_test) = load_data()

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

	# Get k random objects
	for time in range(k):
		i = randint(0, shape[1] - 1)
		j = randint(0, shape[2] - 1)
		objects.append(x[:, i, j, :])

	# Concatenate each pair of objects to form a relation vector
	for i in range(len(objects)):
		for j in range(i, len(objects)):
			relations.append(K.concatenate([objects[i], objects[j]], axis=1))

	# Restack objects into Keras tensor [batch, relation_ID, relation_vectors]
	return K.permute_dimensions(K.stack([r for r in relations], axis=0), [1, 0, 2])

#
# Define CNN Model
#
inputs = Input(shape=input_shape)
x = Conv2D(24, kernel_size=(3, 3), strides=2, activation='relu')(inputs)
x = Conv2D(24, (3, 3), strides=2, activation='relu')(x)
x = Conv2D(24, (3, 3), strides=2, activation='relu')(x)
x = Conv2D(24, (3, 3), strides=2, activation='relu')(x)
shape = K.int_shape(x)

#
# Define Relation Network module
#
RN_inputs = Input(shape=(1, 2 * shape[3]))
RN_x = Dense(256, activation='relu')(RN_inputs)
RN_x = Dense(256, activation='relu')(RN_x)
RN_x = Dense(256, activation='relu')(RN_x)
RN_x = Dropout(.5)(RN_x)
RN_outputs = Dense(256, activation='relu')(RN_x)
RN = Model(inputs=RN_inputs, outputs=RN_outputs)

#
# Implements g_theta.
#
relations = Lambda(getRelationVectors)(x)  # Get tensor [batch, relation_ID, relation_vectors]
g = TimeDistributed(RN)(relations)         # Use TimeDistributed to apply RN on each r in relations.
g = Lambda(lambda x: K.sum(x, axis=1))(g)  # Sum over relation_ID

#
# Define f_phi.
#
f = Dense(256, activation='relu')(g)
f = Dropout(.5)(f)
f = Dense(256, activation='relu')(f)
f = Dropout(.5)(f)
f = Dense(29, activation='relu')(f)
outputs = Dense(num_classes, activation='softmax')(f)

#
# Train model
#
model = Model(inputs=inputs, outputs=outputs)
print(model.summary())
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)