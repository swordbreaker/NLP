from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
import numpy as np
from classification.data_set import DataSet
from classification.ticketing_data import *


def data(root=''):
    """
    Data providing function:

    This function is separated from create_hyperparam_model() so that hyperopt
    won't reload data for each evaluation run.
    """
    labels, class_names = get_merged_labels_three(root='../')

    x = get_doc_vec_ticketing_message(root='../')
    y = labels

    n_values = len(class_names)
    y = np.eye(n_values)[y]
    data_set = DataSet.from_np_array(x, y, class_names=class_names, p_train=0.8, p_val=0.1)

    x_train = data_set.x_train
    y_train = data_set.y_train
    x_val = data_set.x_val
    y_val = data_set.y_val
    x_test = data_set.x_test
    y_test = data_set.y_test

    return x_train, y_train, x_val, y_val, x_test, y_test


def create_hyperparam_model(x_train, y_train, x_val, y_val, x_test, y_test):
    """
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    """
    shape = x_train[0].shape
    model = Sequential()
    model.add(Dense(512, input_shape=shape))
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense({{choice([256, 512, 1024])}}))
    model.add(Activation({{choice(['relu', 'tanh', 'selu'])}}))
    model.add(Dropout({{uniform(0, 1)}}))

    # If we choose 'four', add an additional fourth layer
    if conditional({{choice(['three', 'four'])}}) == 'four':
        model.add(Dense(100))
        model.add(Activation('relu'))

    model.add(Dense(3))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                  optimizer={{choice(['rmsprop', 'adam', 'sgd'])}})

    model.fit(x_train, y_train,
              batch_size={{choice([64, 128])}},
              epochs=20,
              verbose=0,
              validation_data=(x_val, y_val))
    score, acc = model.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


def fit_hyper(root=''):
    best_run, best_model = optim.minimize(model=create_hyperparam_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())
    x_train, y_train, x_val, y_val, x_test, y_test = data(root='')
    print("Evalutation of best performing model:")
    print(best_model.evaluate(x_test, y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
    return best_model
