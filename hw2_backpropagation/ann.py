import gzip
import pickle
import random
import numpy
import math

input_size = 784
hidden_size = 100
output_size = 10
nr_iterations = 50
learning_rate = 0.02
batch_size = 64
dropout = 0.1


def load_data():
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    f.close()
    return train_set, test_set, valid_set


def sigmoid(z):
    try:
        return 1/(1 + math.exp(-z))
    except:
        print(z)
        exit(1)


def sigmoid_derived(z):
    return z*(1-z)


def classify_instance(instance, W, b):
    return numpy.dot(W, instance) + b


def softmax(zs):
    sums = sum([math.exp(z) for z in zs])
    return numpy.array([math.exp(z) / sums for z in zs])


def split_into_batches(train_set, batch_size):
    random.shuffle(train_set)
    batch_list = [train_set[k: k + batch_size] for k in range(0, len(train_set), batch_size)]
    return batch_list


weights = [
    numpy.array([]),
    numpy.array(numpy.random.normal(0., 1. / (math.sqrt(input_size)), (input_size, hidden_size))),
    numpy.array(numpy.random.normal(0., 1. / (math.sqrt(hidden_size)), (hidden_size, output_size)))
]

biases = [
    numpy.array([]),
    numpy.random.normal(0, 1, hidden_size),
    numpy.random.normal(0, 1, output_size)
]

errors_gamma = [
    numpy.array([]),
    numpy.zeros(hidden_size),
    numpy.zeros(output_size)
]

results_z = [
    numpy.array([]),
    numpy.zeros(hidden_size),
    numpy.zeros(output_size)
]

activations_y = [
    numpy.array([]),
    numpy.zeros(hidden_size),
    numpy.zeros(output_size)
]

delta_weights = [
    numpy.array([]),
    numpy.zeros((input_size,hidden_size)),
    numpy.zeros((hidden_size,output_size))
]

delta_bias = [
    numpy.array([]),
    numpy.zeros(hidden_size),
    numpy.zeros(output_size)
]

training_set, testing_set, validation_set = load_data()
training_set = list(zip(*training_set))

for iteration in range(1, nr_iterations+1):
    print("Iteration:", iteration)
    crossent = 0

    dropout_values = numpy.random.binomial(1, 1 - dropout, size=activations_y[1].shape) / (1 - dropout)
    activations_y[1] *= dropout_values

    batches = split_into_batches(training_set, batch_size)
    for batch in batches:
        for x, t in batch:
            # prepare target as vector
            target = numpy.array([1 if int(t == digit) else 0 for digit in range(10)])

            # hidden layer
            results_z[1] = numpy.dot(x, weights[1]) + biases[1]
            activations_y[1] = numpy.array(list(map(lambda x: sigmoid(x), results_z[1])))

            # output layer
            results_z[2] = numpy.dot(activations_y[1], weights[2]) + biases[2]
            activations_y[2] = softmax(results_z[2])

            # error last layer
            errors_gamma[2] = activations_y[2] - target

            # backpropagate the error
            # error hidden layer
            errors_gamma[1] = numpy.dot(weights[2], errors_gamma[2]) * numpy.array(list(map(lambda x: sigmoid_derived(x), activations_y[1])))

            # delta weight and bias
            delta_weights[2] += numpy.outer(activations_y[1], errors_gamma[2])
            delta_bias[2] += errors_gamma[2]

            delta_weights[1] += numpy.outer(x, errors_gamma[1])
            delta_bias[1] += errors_gamma[1]

            # update crossentropy
            crossentropy = [-math.log(activations_y[2][i]) if target[i] == 1 else -math.log(
                1 - activations_y[2][i]) for i in range(10)]
            crossent += sum(crossentropy) / 50000

        # update weights and biases
        weights -= numpy.array(delta_weights) * learning_rate
        biases -= numpy.array(delta_bias) * learning_rate

        #reset deltas
        delta_weights = [
            numpy.array([]),
            numpy.zeros((input_size, hidden_size)),
            numpy.zeros((hidden_size, output_size))
        ]

        delta_bias = [
            numpy.array([]),
            numpy.zeros(hidden_size),
            numpy.zeros(output_size)
        ]

    print("Crossentropy:", crossent)

    # Validation
    correct = 0
    for x, t in list(zip(*validation_set)):
        # prepare target as vector
        target = numpy.array([1 if int(t == digit) else 0 for digit in range(10)])

        # hidden layer
        results_z[1] = numpy.dot(x, weights[1]) + biases[1]
        activations_y[1] = numpy.array(list(map(lambda x: sigmoid(x), results_z[1])))

        # output layer
        results_z[2] = numpy.dot(activations_y[1], weights[2]) + biases[2]
        activations_y[2] = softmax(results_z[2])

        index_max = activations_y[2].argmax()
        if t == index_max:
            correct += 1
    print("Correct:", correct)
    print("Precision :", correct / len(list(zip(*validation_set))) * 100)
print("End of training")

# Test
correct = 0
for x, t in list(zip(*testing_set)):
    # prepare target as vector
    target = numpy.array([1 if int(t == digit) else 0 for digit in range(10)])

    # hidden layer
    results_z[1] = numpy.dot(x, weights[1]) + biases[1]
    activations_y[1] = numpy.array(list(map(lambda x: sigmoid(x), results_z[1])))

    # output layer
    results_z[2] = numpy.dot(activations_y[1], weights[2]) + biases[2]
    activations_y[2] = softmax(results_z[2])

    index_max = activations_y[2].argmax()
    if t == index_max:
        correct += 1
print("Correct:", correct)
print("Precision :", correct / 10000 * 100)
