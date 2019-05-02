import keras
from keras.datasets import cifar10
from keras import Sequential
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import *
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras_preprocessing.image import ImageDataGenerator


def load_and_preprocess_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return x_train, y_train, x_test, y_test


def construct_model():
    number_of_classes = 10
    weight_decay = 1e-4

    model = Sequential()
    model.add(Lambda(lambda x: x / 255, input_shape=x_train.shape[1:]))
    model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),
                     kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=1 / 5.5))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),
                     kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=1 / 5.5))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),
                     kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=1 / 5.5))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),
                     kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=1 / 5.5))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),
                     kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=1 / 5.5))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),
                     kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=1 / 5.5))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(200, kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=1 / 5.5))
    model.add(Dense(number_of_classes, activation='softmax', kernel_initializer='he_normal'))

    opt_rms = keras.optimizers.rmsprop(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=opt_rms, metrics=['accuracy'])

    model.summary()
    return model


def plotting(history):
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


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_and_preprocess_data()
    model = construct_model()

    train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)

    batch_size = 64
    train_generator = train_datagen.flow(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        shuffle=True)

    checkpointer = ModelCheckpoint(
        filepath="models/{epoch:02d}-{val_acc:.4f}-{val_loss:.4f}.hd5",
        save_weights_only=False, verbose=1, save_best_only=False)

    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=5, min_lr=0.0003)

    history = model.fit_generator(train_generator,
                                  epochs=200,
                                  steps_per_epoch=len(x_train) / batch_size,
                                  validation_data=(x_test, y_test),
                                  shuffle=True,
                                  callbacks=[checkpointer, reduce_lr])

    scores = model.evaluate(x_test, y_test)
    model.summary()
    print('Loss: %.3f' % scores[0])
    print('Accuracy: %.3f' % scores[1])

    plotting(history)
