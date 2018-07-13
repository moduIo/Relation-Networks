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

from __future__ import print_function
import tensorflow as tf
import numpy as np
import keras
from keras.datasets import mnist
from keras.optimizers import Adam
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, Dropout, Reshape, Lambda, Conv2D, MaxPooling2D, TimeDistributed
from keras.models import Model
from random import *

#
# Preprocesses the input image by cropping and random rotations.
#
def process_image(x):
	target_height, target_width = 128, 128
	rotation_range = .05  # In radians
	degs = random.uniform(-rotation_range, rotation_range)

	x = tf.image.resize_image_with_crop_or_pad(x, target_height, target_width)
	x = tf.image.rotate(x, degs)

	return x

#
# Returns relation vectors from an input convolution tensor map.
# A relation vector is the concatenation of two objects, 
#     in this case the objects are "pixels" of the tensor.
#
def get_relation_vectors(x):
	objects = []
	relations = []
	shape = K.int_shape(x)
	k = 25     # Hyperparameter which controls how many objects are considered
	keys = []

	# Get k unique random objects
	while k > 0:
		i = randint(0, shape[1] - 1)
		j = randint(0, shape[2] - 1)

		if not (i, j) in keys:
			keys.append((i, j))
			objects.append(x[:, i, j, :])
			k -= 1

	# Concatenate each pair of objects to form a relation vector
	for i in range(len(objects)):
		for j in range(i, len(objects)):
			relations.append(K.concatenate([objects[i], objects[j]], axis=1))

	# Restack objects into Keras tensor [batch, relation_ID, relation_vectors]
	return K.permute_dimensions(K.stack([r for r in relations], axis=0), [1, 0, 2])

#
# Environment Parameters
#
batch_size = 64
epochs = 12
learning_rate = .00025
img_rows, img_cols = 28, 28  # input image dimensions
num_classes = 10

#
# Load & Preprocess MNIST
#
(x_train, y_train), (x_test, y_test) = mnist.load_data()
y_train = keras.utils.to_categorical(y_train, num_classes) # convert class vectors to binary class matrices
y_test = keras.utils.to_categorical(y_test, num_classes)

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    image_input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    image_input_shape = (img_rows, img_cols, 1)

#
# Define CNN
#
image_inputs = Input(shape=image_input_shape)
x = Lambda(process_image)(image_inputs)
x = Conv2D(24, kernel_size=(3, 3), strides=2, activation='relu')(x)
x = Conv2D(24, kernel_size=(3, 3), strides=2, activation='relu')(x)
x = Conv2D(24, kernel_size=(3, 3), strides=2, activation='relu')(x)
x = Conv2D(24, kernel_size=(3, 3), strides=2, activation='relu')(x)
shape = K.int_shape(x)

#
# Define LSTM
#
text_inputs = Input(shape=text_input_shape)

#
# Define Relation Network layer
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
relations = Lambda(get_relation_vectors)(x)  # Get tensor [batch, relation_ID, relation_vectors]
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

model.compile(optimizer=Adam(lr=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)