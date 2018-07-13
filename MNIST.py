from __future__ import print_function
import keras
import numpy as np
from keras.datasets import mnist
from keras import backend as K
from keras.layers import Input, Dense, Reshape, Lambda, Add, Conv2D, TimeDistributed
from keras.models import Model

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
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
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#
#
#
def getRelationVectors(x):
	objects = []
	relations = []
	shape = K.int_shape(x)

	# Get each object
	for i in range(5):
		for j in range(5):
			objects.append(x[:,i,j,:])

	# Concatenate each pair of objects to form a relation
	for i in range(len(objects)):
		for j in range(i, len(objects)):
			relations.append(K.concatenate([objects[i], objects[j]], axis=1))

	# Restack objects into tensor [batch, relation_ID, relation_vectors]
	return K.permute_dimensions(K.stack([r for r in relations], axis=0), [1, 0, 2])

#
# Define model
#
inputs = Input(shape=input_shape)
x = Conv2D(4, kernel_size=(3, 3), activation='relu')(inputs)
x = Conv2D(8, (3, 3), activation='relu')(x)

shape = K.int_shape(x)
print(x)
print(shape)

x = Lambda(getRelationVectors)(x)
print(x)
print(K.int_shape(x))

#
# Define RN layer
#
RN_inputs = Input(shape=(1,2 * shape[3]))
RN_x = Dense(8, activation='relu')(RN_inputs)
RN_outputs = Dense(20, activation='relu')(RN_x)
RN = Model(inputs=RN_inputs, outputs=RN_outputs)

x = TimeDistributed(RN)(x)

print(x)
print(K.int_shape(x))
x = Lambda(lambda x: K.sum(x, axis=1))(x)

print(x)
print(K.int_shape(x))
outputs = Dense(10, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)
print(model.summary())
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)