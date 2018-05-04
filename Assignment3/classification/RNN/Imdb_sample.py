import numpy as np
from keras_rnn import run_rnn
import keras
from keras.datasets import imdb
import time
import datetime


max_len = 200
n_outputs = 2
num_words = 1000

def preprocess(x: np.ndarray, y: np.ndarray):
    x = keras.preprocessing.sequence.pad_sequences(x, maxlen=max_len, padding='pre', value=0)
    x = x.astype('float32') / num_words
    # x = keras.utils.to_categorical(x)
    print(x)

    y = keras.utils.to_categorical(y, num_classes=n_outputs)
    x_shaped = x.reshape(x.shape[0], x.shape[1], 1)
    return x_shaped, y


(x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = imdb.load_data(path="imdb.npz",
                                                                      num_words=num_words,
                                                                      maxlen=max_len,
                                                                      seed=113,
                                                                      start_char=1,
                                                                      oov_char=2,
                                                                      index_from=3,
                                                                      skip_top=0)


x_train, y_train = preprocess(x_train_raw, y_train_raw)

print()
print('X_train:', x_train.shape)
print('y_train:', y_train.shape)

epochs = 100
model = run_rnn(n_outputs, x_train, y_train, epochs=epochs, learning_rate=0.001)

t = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')

model.save(f"imdb_{epochs}_{t}.h5")
# to load: keras.models.load_model(filepath)
