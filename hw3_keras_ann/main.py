from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images, extract_labels
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, InputLayer, Flatten, Lambda
import matplotlib.pyplot as plt
import keras


def load_data():
    return extract_images(open('data/emnist-bymerge-train-images-idx3-ubyte.gz', 'rb')), \
           extract_labels(open('data/emnist-bymerge-train-labels-idx1-ubyte.gz', 'rb')), \
           extract_images(open('data/emnist-bymerge-test-images-idx3-ubyte.gz', 'rb')), \
           extract_labels(open('data/emnist-bymerge-test-labels-idx1-ubyte.gz', 'rb'))


def construct_model():
    model = Sequential()
    model.add(InputLayer((28, 28, 1)))
    model.add(Lambda(lambda x: x/255))
    model.add(Flatten())
    model.add(Dense(800, activation='relu', kernel_initializer='lecun_normal'))
    model.add(Dropout(0.1))
    model.add(Dense(300, activation='relu', kernel_initializer='lecun_normal'))
    model.add(Dropout(0.2))
    model.add(Dense(47, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer="sgd", metrics=['acc'])
    return model

def training(model):
    train_datagen = ImageDataGenerator(
        shear_range=0.2
    )

    train_generator = train_datagen.flow(
        x=train_x,
        y=train_y,
        batch_size=32,
        shuffle=True)

    early_stopping_monitor = EarlyStopping(patience=50)
    checkpointer = keras.callbacks.ModelCheckpoint(filepath="models/{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}.h5", save_weights_only=False, verbose=1, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2,
                                   patience=5, min_lr=0.001)

    history = model.fit_generator(train_generator,
                        epochs=100,
                        steps_per_epoch=len(train_x) / 32,
                        validation_data=(test_x, test_y),
                        callbacks=[checkpointer, early_stopping_monitor, reduce_lr])
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


def preprocessing():
    train_x, train_y1, test_x, test_y1 = load_data()
    train_y = keras.utils.to_categorical(train_y1, 47)
    targets = np.eye(47)
    train_y, test_y = [targets[y] for y in [train_y1, test_y1]]
    return train_x, train_y, test_x, test_y


def load_model_from_file():
    model = load_model(filepath="./model.h5")
    train_x, train_y, test_x, test_y = load_data()
    loss, acc = model.evaluate(test_x, test_y)
    print(loss, acc)


if __name__ == '__main__':
    train_x, train_y, test_x, test_y = preprocessing()
    model = construct_model()
    model.summary()
    training(model)


