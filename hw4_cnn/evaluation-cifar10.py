from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import load_model


if __name__ == '__main__':
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
    model = load_model("./model.h5")
    scores = model.evaluate(X_test, to_categorical(Y_test))
    model.summary()
    print('Loss: %.3f' % scores[0])
    print('Accuracy: %.3f' % scores[1])