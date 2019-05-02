from keras.utils import np_utils
import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, Dropout, InputLayer, Flatten, Lambda
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from mnist import MNIST

def load_data(path, ):
    mndata = MNIST(path)

    X_train, y_train = mndata.load(path + '/emnist-bymerge-train-images-idx3-ubyte',
                                   path + '/emnist-bymerge-train-labels-idx1-ubyte')
    X_test, y_test = mndata.load(path + '/emnist-bymerge-test-images-idx3-ubyte',
                                 path + '/emnist-bymerge-test-labels-idx1-ubyte')


    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    X_train = normalize(X_train)
    X_test = normalize(X_test)

    y_train = preprocess_labels(y_train, 47)
    y_test = preprocess_labels(y_test, 47)

    return X_train, y_train, X_test, y_test


def normalize(array):
    array = array.astype('float32')
    array /= 255

    return array


def preprocess_labels(array, nb_classes):
    return np_utils.to_categorical(array, nb_classes)


def construct_model():
    model = Sequential()
    model.add(Dense(800, input_shape=(784,), activation='relu', kernel_initializer='lecun_normal', bias_initializer='zeros'))
    model.add(Dropout(0.1))
    model.add(Dense(300, activation='relu', kernel_initializer='lecun_normal', bias_initializer='zeros'))
    model.add(Dropout(0.2))
    model.add(Dense(47, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer="sgd", metrics=['acc'])
    return model


def training(model, train_x, train_y, test_x, test_y):
    early_stopping_monitor = EarlyStopping(patience=50)
    checkpointer = keras.callbacks.ModelCheckpoint(filepath="models/model_-{val_acc:.4f}.h5",
                                                   save_weights_only=False, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2,
                                  patience=5, min_lr=0.001)

    history = model.fit(train_x, train_y, epochs=56, validation_data=(test_x, test_y), batch_size=32, callbacks=[checkpointer, early_stopping_monitor, reduce_lr])
    plotting(history)


def plotting(history):
    print(history.history.keys())

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()



def load_model_from_file():
    model= load_model(filepath="models\\model.h5")
    train_x, train_y1, test_x, test_y1 = load_data("data")
    loss, acc = model.evaluate(test_x, test_y)
    print(loss, acc)


if __name__ == '__main__':
    train_x, train_y, test_x, test_y = load_data("data")
    load_model_from_file()

