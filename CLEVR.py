#
# Implementation of Relation Networks using CLEVR.
# https://arxiv.org/pdf/1706.01427.pdf
#
from __future__ import print_function
import tensorflow as tf
import numpy as np
import keras
from keras.optimizers import Adam
from keras import backend as K
from keras.layers import Input, Dense, Dropout, Reshape, Lambda, Embedding, LSTM, Conv2D, MaxPooling2D, TimeDistributed, RepeatVector, Concatenate
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from random import *
import json
import os.path

#
# Loads CLEVR dataset.
#
def load_data(split, n, vocab_size):
	path = '../../Datasets/CLEVR_v1.0'
	questions_path = path + '/questions/CLEVR_' + split + '_questions.json'
	subset_questions_path = path + '/questions/CLEVR_' + split + '_questions_' + str(n) + '.json'
	images_path = path + '/images'
	x_text = []
	x_image = []
	y = []
	labels = {}
	num_labels = 0

	# Attempt to load saved JSON subset of the questions
	if os.path.exists(subset_questions_path):
		with open(subset_questions_path) as f:
			data = json.load(f)
	else:
		with open(questions_path) as f:
			data = json.load(f)

		with open(subset_questions_path, 'w') as outfile:
			json.dump(data['questions'][0:n], outfile)

		print('JSON subset saved to file...')

	print('Data loaded...')

	# Process data
	for q in data[0:n]:
		# Create an index for each answer
		if not q['answer'] in labels:
			labels[q['answer']] = num_labels
			num_labels += 1

		x_text.append(q['question'])
		y.append(labels[q['answer']])

	# Convert question corpus into sequential encoding for LSTM
	t = Tokenizer(num_words=vocab_size)
	t.fit_on_texts(x_text)
	sequences = t.texts_to_sequences(x_text)
	x_text = sequence.pad_sequences(sequences, maxlen=vocab_size)

	# Convert labels to categorical labels
	y = keras.utils.to_categorical(y, num_labels + 1)

	print(x_text.shape)
	print(y.shape)
	exit()

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
vocab_size = 1000

#
# Load & Preprocess CLEVR
#
(x_train, y_train), num_classes = load_data('train', 2000, vocab_size)

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    image_input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    image_input_shape = (img_rows, img_cols, 1)

#
# Define LSTM
#
text_inputs = Input(shape=text_input_shape)
text_x = Embedding(max_text_features, 128)(text_inputs)
text_x = LSTM(128)(text_x)

#
# Define CNN
#
image_inputs = Input(shape=image_input_shape)
image_x = Lambda(process_image)(image_inputs)
image_x = Conv2D(24, kernel_size=(3, 3), strides=2, activation='relu')(image_x)
image_x = Conv2D(24, kernel_size=(3, 3), strides=2, activation='relu')(image_x)
image_x = Conv2D(24, kernel_size=(3, 3), strides=2, activation='relu')(image_x)
image_x = Conv2D(24, kernel_size=(3, 3), strides=2, activation='relu')(image_x)
shape = K.int_shape(image_x)

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
# Implements g_theta
#
relations = Lambda(get_relation_vectors)(image_x)           # Get tensor [batch, relation_ID, relation_vectors]
question = RepeatVector(K.int_shape(relations)[1])(text_x)  # Shape question vector to same size as relations
relations = Concatenate(axis=1)([relations, question])      # Merge tensors [batch, relation_ID, relation_vectors, question_vector]
g = TimeDistributed(RN)(relations)                          # TimeDistributed applies RN to relation vectors.
g = Lambda(lambda x: K.sum(x, axis=1))(g)                   # Sum over relation_ID

#
# Define f_phi
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