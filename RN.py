import keras
from keras import backend as K
from keras.layers import Input, Dense, Reshape, Lambda, Add, TimeDistributed
from keras.models import Model

epochs = 1
batch_size = 8

#
# Generate dummy data
#
import numpy as np
x_train = np.random.random((1000, 32))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
x_test = np.random.random((100, 32))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

#
# Define RN layer
#
RN_inputs = Input(shape=(1,4))
x = Dense(8, activation='relu')(RN_inputs)
RN_outputs = Dense(5, activation='relu')(x)
RN = Model(inputs=RN_inputs, outputs=RN_outputs)

#
# Define network architecture
#
inputs = Input(shape=(32,))
x = Reshape((8,4))(inputs)
x = TimeDistributed(RN)(x)
x = Lambda(lambda x: K.sum(x, axis=1))(x)
outputs = Dense(10, activation='softmax')(x)

#
# Define model
#
model = Model(inputs=inputs, outputs=outputs)
print model.summary()
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)