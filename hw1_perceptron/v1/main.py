import gzip
import pickle
from tema1.perceptron_v1 import Perceptron


def load_data():
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_set, validation_set, test_set = pickle.load(f, encoding='latin1')
    f.close()
    return training_set, test_set


if __name__ == '__main__':
    number_of_iterations = 10
    batch_size = 64
    learning_rate = 0.0005

    training_set, test_set = load_data()
    perceptrons = []
    for digit in range(10):
        perceptron = Perceptron(digit, training_set, number_of_iterations, batch_size, learning_rate)
        perceptron.train()
        perceptrons.append(perceptron)

    print("End of training")

    correct = 0
    for x, t in list(zip(*test_set)):
        results = []
        for perceptron in perceptrons:
            result = perceptron.classify_instance(x)
            results.append(result[0])
        max_result = max(results)
        index_max = results.index(max_result)
        if t == index_max:
            correct += 1
    print("Correct:", correct)
    print("Precision :", correct / 10000 * 100)



