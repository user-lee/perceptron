
import array
import random

ISIZE = 2
WSIZE = ISIZE + 1
LEARNING_RATE = 0.1
ITERATIONS = 10

weights = array.array('f', [0, 0, 0])


def initialize():

    for i in range(WSIZE):
        weights[i] = random.random()


def feed_forward(inputs):

    sum_ = 0.0
    for i in range(ISIZE):
        sum_ += weights[i] * inputs[i]

    sum_ += weights[ISIZE]

    return 1 if sum_ >= 1.0 else 0


def train():

    iteration_error = 1
    iterations = 0

    test = [[0, 0], [0, 1], [1, 0], [1, 1]]

    while iteration_error > 0 and iterations < ITERATIONS:

        print("iterations:", iterations)

        for i in range(len(test)):
            desired_output = test[i][0] or test[i][0]
            output = feed_forward(test[i])
            error = desired_output - output
            weights[0] += LEARNING_RATE * error * test[i][0]
            weights[1] += LEARNING_RATE * error * test[i][1]
            weights[2] += LEARNING_RATE * error
            iteration_error += error * error
            print(test[i][0], test[i][1], desired_output, output, iteration_error, sep=',')

        print("weights:", weights)
        iterations += 1


if __name__ == '__main__':

    train()
