import numpy as np
from keras_rnn import run_rnn
import keras

from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
nb_classes = 10

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# one-hot encoding:
Y_train = keras.utils.to_categorical(y_train, num_classes=nb_classes)
Y_test = keras.utils.to_categorical(y_test, num_classes=nb_classes)

print()
print('MNIST data loaded: train:', len(X_train), 'test:', len(X_test))
print('X_train:', X_train.shape)
print('y_train:', Y_train.shape)

# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# y_train = keras.utils.to_categorical(y_train)

run_rnn(10, X_train, Y_train, 28)

