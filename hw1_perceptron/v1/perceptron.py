import random
import numpy


class Perceptron:
    def __init__(self, digit, training_set, number_of_iterations, batch_size, learning_rate):
        self.W = numpy.random.rand(1, 784)
        self.b = random.random()
        self.digit = digit
        self.training_set = list(zip(*training_set))
        self.number_of_iterations = number_of_iterations
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def train(self):
        for it in range(self.number_of_iterations):
            print("Iteration:", it , ", For digit:", self.digit)
            batch_counter = 1
            W_update = numpy.zeros((1,784))
            b_update = 0
            for x, t in self.training_set:
                t = int(t == self.digit)
                z = numpy.dot(self.W, x) + self.b
                y = self.activation(z)
                W_update = W_update + numpy.multiply(numpy.multiply((t-y), x), self.learning_rate)
                b_update = b_update + numpy.multiply((t-y), self.learning_rate)
                if batch_counter == self.batch_size:
                    # update
                    self.W += W_update
                    self.b += b_update
                    # reset
                    batch_counter = 0
                    W_update = numpy.zeros((1, 784))
                    b_update = 0
                batch_counter += 1
            # if the batches don't fit the training set (exact division)
            if batch_counter != 1:
                # update
                self.W += W_update
                self.b += b_update

    def activation(self, z):
        if z > 0:
            return 1
        return 0

    def classify_instance(self, instance):
        return numpy.dot(self.W, instance) + self.b
