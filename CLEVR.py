#
# Implementation of Relation Networks using CLEVR.
# https://arxiv.org/pdf/1706.01427.pdf
#
from __future__ import print_function
import json
import os.path
import random as ra
import tensorflow as tf
import numpy as np
import keras
from keras.optimizers import Adam
from keras import backend as K
from keras.layers import Input, Dense, Dropout, BatchNormalization, Reshape, Lambda, Embedding, LSTM, Conv2D, MaxPooling2D, TimeDistributed, RepeatVector, Concatenate
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint, TensorBoard
from scipy import ndimage, misc

#
# Loads & Preprocesses CLEVR dataset.
#
def load_data(split, n, vocab_size, tokenizer=None):
	# Dataset paths
	path = '../../Datasets/CLEVR_v1.0'
	questions_path = path + '/questions/CLEVR_' + split + '_questions.json'
	subset_questions_path = path + '/questions/CLEVR_' + split + '_questions_' + str(n) + '.json'
	images_path = path + '/images/' + split + '/'
	
	x_text = []     # List of questions
	x_image = []    # List of images
	y = []          # List of answers
	num_labels = 0  # Current number of labels, used to create index mapping
	labels = {}     # Dictionary mapping of ints to labels
	images = {}     # Dictionary of images, to minimize number of imread ops

	# Attempt to load saved JSON subset of the questions
	print('Loading data...')

	if os.path.exists(subset_questions_path):
		with open(subset_questions_path) as f:
			data = json.load(f)
	else:
		with open(questions_path) as f:
			data = json.load(f)

		data = data['questions'][0:n]

		with open(subset_questions_path, 'w') as outfile:
			json.dump(data, outfile)

		print('JSON subset saved to file...')

	# Store data
	print('Storing data...')

	for q in data[0:n]:
		# Create an index for each answer
		if not q['answer'] in labels:
			labels[q['answer']] = num_labels
			num_labels += 1

		# Create an index for each image
		if not q['image_filename'] in images:
			images[q['image_filename']] = misc.imread(images_path + q['image_filename'], mode='RGB')

		x_text.append(q['question'])
		x_image.append(images[q['image_filename']])
		y.append(labels[q['answer']])

	# Convert question corpus into sequential encoding for LSTM
	print('Processing data...')

	if not tokenizer:
		tokenizer = Tokenizer(num_words=vocab_size)

	tokenizer.fit_on_texts(x_text)
	sequences = tokenizer.texts_to_sequences(x_text)
	x_text = sequence.pad_sequences(sequences, maxlen=vocab_size)

	# Convert x_image to np array
	x_image = np.array(x_image)

	# Convert labels to categorical labels
	y = keras.utils.to_categorical(y, num_labels)

	print('Text: ', x_text.shape)
	print('Image: ', x_image.shape)
	print('Labels: ', y.shape)

	return ([x_text, x_image], y), num_labels, tokenizer

#
# Preprocesses the input image by cropping and random rotations.
#
def process_image(x):
	target_height, target_width = 128, 128
	rotation_range = .05  # In radians
	degs = ra.uniform(-rotation_range, rotation_range)

	x = tf.image.resize_images(x, (target_height, target_width), method=tf.image.ResizeMethod.AREA)
	x = tf.contrib.image.rotate(x, degs)

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
		i = ra.randint(0, shape[1] - 1)
		j = ra.randint(0, shape[2] - 1)

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
samples = 60000
epochs = 100
batch_size = 64
learning_rate = .00025
vocab_size = 1024
img_rows, img_cols = 320, 480
image_input_shape = (img_rows, img_cols, 3)

#
# Load & Preprocess CLEVR
#
(x_train, y_train), num_labels, tokenizer = load_data('train', samples, vocab_size)

#
# Define LSTM
#
text_inputs = Input(shape=(vocab_size,), name='text_input')
text_x = Embedding(vocab_size, 128)(text_inputs)
text_x = LSTM(128)(text_x)

#
# Define CNN
#
image_inputs = Input(shape=image_input_shape, name='image_input')
image_x = Lambda(process_image)(image_inputs)
image_x = Conv2D(24, kernel_size=(3, 3), strides=2, activation='relu')(image_x)
image_x = BatchNormalization()(image_x)
image_x = Conv2D(24, kernel_size=(3, 3), strides=2, activation='relu')(image_x)
image_x = BatchNormalization()(image_x)
image_x = Conv2D(24, kernel_size=(3, 3), strides=2, activation='relu')(image_x)
image_x = BatchNormalization()(image_x)
image_x = Conv2D(24, kernel_size=(3, 3), strides=2, activation='relu')(image_x)
image_x = BatchNormalization()(image_x)
shape = K.int_shape(image_x)

#
# Define Relation Network layer
#
RN_inputs = Input(shape=(1, (2 * shape[3]) + K.int_shape(text_x)[1]))
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
relations = Concatenate(axis=2)([relations, question])      # Merge tensors [batch, relation_ID, relation_vectors, question_vector]
g = TimeDistributed(RN)(relations)                          # TimeDistributed applies RN to relation vectors.
g = Lambda(lambda x: K.sum(x, axis=1))(g)                   # Sum over relation_ID

#
# Define f_phi
#
f = Dense(256, activation='relu')(g)
f = Dropout(.5)(f)
f = Dense(256, activation='relu')(f)
f = Dropout(.5)(f)
outputs = Dense(num_labels, activation='softmax')(f)

#
# Train model
#
model = Model(inputs=[text_inputs, image_inputs], outputs=outputs)
print(model.summary())

model.compile(optimizer=Adam(lr=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, 
	      batch_size=batch_size, 
	      epochs=epochs, 
	      shuffle=True,
          callbacks=[ModelCheckpoint('models/' + str(samples) + '.hdf5', period=1),
                     TensorBoard(log_dir='logs/' + str(samples))])