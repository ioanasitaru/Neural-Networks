import gzip
import pickle
import numpy


def load_data():
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_set, validation_set, test_set = pickle.load(f, encoding='latin1')
    f.close()
    return training_set, test_set


def activation(z):
    if z > 0:
        return 1
    return 0


def classify_instance(instance, W, b):
    return numpy.dot(W, instance) + b


if __name__ == '__main__':
    number_of_iterations = 10
    batch_size = 64
    learning_rate = 0.001
    input_size = 784

    training_set, test_set = load_data()

    b = numpy.random.rand(10)
    W = numpy.random.rand(10, input_size)

    # lista de tuple
    training_set = list(zip(*training_set))

    for it in range(number_of_iterations):
        print("Iteration:", it)
        batch_counter = 1
        W_update = numpy.zeros((10, input_size))
        b_update = numpy.zeros(10)

        numpy.random.shuffle(training_set)
        for x, t in training_set:
            t = numpy.array([1 if int(t == digit) else 0 for digit in range(10)])
            z = W.dot(x) + b
            y = numpy.array([activation(z_i) for z_i in z])

            W_update = W_update + numpy.outer((t-y)*learning_rate, x)
            b_update = b_update + (t-y)*learning_rate

            if batch_counter == batch_size:
                # update
                W += W_update
                b += b_update
                # reset
                batch_counter = 0
                W_update = numpy.zeros((10, input_size))
                b_update = numpy.zeros(10)
            batch_counter += 1
        # if the batches don't fit the training set (exact division)
        if batch_counter != 1:
            # update
            W += W_update
            b += b_update

    print("End of training")

    correct = 0
    for x, t in list(zip(*test_set)):
        results = classify_instance(x, W, b)
        max_result = max(results)
        index_max = numpy.where(results == max_result)
        if t == index_max:
            correct += 1
    print("Correct:", correct)
    print("Precision :", correct / 10000 * 100)