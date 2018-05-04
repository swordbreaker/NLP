from keras.models import Sequential
from keras.layers import Dense, Activation, SimpleRNN, GRU, RNN, Masking, Dropout
import keras
import numpy as np
import matplotlib.pyplot as plt


def run_rnn(n_outputs: int, x_train: np.ndarray, y_train: np.ndarray, n_features=1, epochs=10, learning_rate=0.001):
    """
    :param learning_rate:
    :param n_outputs: number of categories
    :param x_train: in the format [samples, time steps, features]
    :param y_train: one hot encoded with keras.utils.to_categorical
    :param n_features: number of feature
    :param epochs:
    :return: void
    """

    n_neurons = 100

    cells = [
        keras.layers.LSTMCell(n_neurons),
        keras.layers.LSTMCell(n_neurons),
        keras.layers.LSTMCell(n_neurons),
    ]

    model = Sequential([
        # RNN(cells, input_shape=(None, n_features)),
        #GRU(n_neurons, input_shape=(None, n_features)),
        Masking(mask_value=0, input_shape=(None, n_features)),
        # SimpleRNN(n_neurons, input_shape=(None, n_features), return_state=True),
        # SimpleRNN(n_neurons),
        GRU(n_neurons),
        Dropout(0.5),
        Dense(n_outputs),
        Activation('softmax')
    ])

    adam = keras.optimizers.adam(lr=learning_rate)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    board_callback = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, batch_size=32, write_graph=True, write_grads=True,
                                write_images=True, embeddings_freq=0, embeddings_layer_names=True,
                                embeddings_metadata=True)

    print(model.summary())

    history = model.fit(x_train, y_train, epochs=epochs, batch_size=100, validation_split=0.33, verbose=2, callbacks=[board_callback])

    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    score = model.evaluate(x_train, y_train, batch_size=128)

    print(score)

    return model