import math
import array
import random
from dataset import dataset as train_data

INP_NEURONS = 4
HID_NEURONS = 25
OUT_NEURONS = 3

LEARNING_RATE = 0.05

inputs = array.array('d', [1.0] * (INP_NEURONS + 1))
hidden = array.array('d', [1.0] * (HID_NEURONS + 1))
outputs = array.array('d', [0.0] * OUT_NEURONS)

# why subtract 0.5
weights_hidden_input = [array.array('d', [random.random() - 0.5 for i in range(INP_NEURONS + 1)]) for j in range(HID_NEURONS)]
weights_output_hidden = [array.array('d', [random.random() - 0.5 for i in range(HID_NEURONS + 1)]) for j in range(OUT_NEURONS)]


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def sigmoid_d(x):
    return x * (1.0 - x)


def nn_feed_forward():
    for i in range(HID_NEURONS):
        hidden[i] = 0.0
        for j in range(INP_NEURONS + 1):
            hidden[i] += weights_hidden_input[i][j] * inputs[j]
        hidden[i] = sigmoid(hidden[i])

    for i in range(OUT_NEURONS):
        outputs[i] = 0.0
        for j in range(HID_NEURONS + 1):
            outputs[i] += weights_output_hidden[i][j] * hidden[j]
        outputs[i] = sigmoid(outputs[i])

    best = 0
    _max = outputs[0]

    for i in range(1, OUT_NEURONS):
        if outputs[i] > _max:
            best = i
            _max = outputs[i]

    return best


def nn_back_progagate(test):

    err_out = array.array('d', [0.0] * OUT_NEURONS)
    err_hid = array.array('d', [0.0] * HID_NEURONS)

    for out in range(OUT_NEURONS):
        err_out[out] = (train_data[test][1][out] - outputs[out]) * sigmoid_d(outputs[out])

    for hid in range(HID_NEURONS):
        for out in range(OUT_NEURONS):
            err_hid[hid] += err_out[out] * weights_output_hidden[out][hid]
        err_hid[hid] *= sigmoid_d(hidden[hid])

    for out in range(OUT_NEURONS):
        for hid in range(HID_NEURONS):
            weights_output_hidden[out][hid] += LEARNING_RATE * err_out[out] * hidden[hid]

    for hid in range(HID_NEURONS):
        for inp in range(INP_NEURONS + 1):
            weights_hidden_input[hid][inp] += LEARNING_RATE * err_hid[hid] * inputs[inp]


def nn_set_inputs(test):
    for i in range(INP_NEURONS):
        inputs[i] = train_data[test][0][i]


def nn_train(iterations):

    for i in range(iterations):
        test = random.randint(0, len(train_data)-1)
        nn_set_inputs(test)
        nn_feed_forward()
        nn_back_progagate(test)


def nn_test(tests):

    for i in range(tests):
        test = random.randint(1, len(train_data)-1)
        nn_set_inputs(test)
        result = nn_feed_forward()
        print("Test %d classified as %d (%g %g %g)" %
              (test, result, train_data[test][1][0], train_data[test][1][1], train_data[test][1][2]))


if __name__ == '__main__':

    nn_train(30000)
    nn_test(10)
