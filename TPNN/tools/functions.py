import math

import numpy as np
import numpy.linalg as la


# activation functions
from TPNN.tools.CNNarchitecture import Net


def ident(x):
    return x


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def th(x):
    return np.tanh(x)


def ELU(x, alpha):
    if x >= 0:
        return x
    else:
        return alpha * (np.exp(x) - 1)


# derivatives of activation functions
def ident_der(x):
    return 1


def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x))


def th_der(x):
    return 1 - th(x) ** 2


def ELU_der(x, alpha):
    if x >= 0:
        return 1
    else:
        return alpha * np.exp(x)


def get_der(function_ident):
    if function_ident == sigmoid:
        return sigmoid_der
    elif function_ident == ident:
        return ident_der
    elif function_ident == th:
        return th_der
    elif function_ident == ELU:
        return ELU_der
    else:
        assert False

#--------- loss functions:

# checks is vector an one-hot-encoding vector
def check_one_hot_encoding(one_hot_enc_vector):
    assert np.max(one_hot_enc_vector) == 1 and np.sum(one_hot_enc_vector) == 1


# get idx od 1 in one-hot-encoding vector
def get_ones_pos(one_hot_enc_vector):
    check_one_hot_encoding(one_hot_enc_vector)

    for i in range(len(one_hot_enc_vector)):
        if one_hot_enc_vector[i] == 1:
            return i

    assert False


# logarithmic loss function
def log_loss(one_hot_enc_vector, result_vector):
    assert len(one_hot_enc_vector.shape) == len(result_vector.shape) == 1
    assert one_hot_enc_vector.shape == result_vector.shape
    check_one_hot_encoding(one_hot_enc_vector)

    return -np.sum(np.log(result_vector) * one_hot_enc_vector)


# gets loss function gradient vector with respect to predicted vector
def der_loss_y(loss_function, predicted_vector, actual_vector):
    assert predicted_vector.shape == actual_vector.shape
    assert len(predicted_vector.shape) == len(actual_vector.shape) == 1

    result = np.zeros(actual_vector.shape)

    # gets gradient vector for each type of loss function
    if loss_function == log_loss:
        idx = get_ones_pos(actual_vector)
        result[idx] = 1 / predicted_vector[idx]

        return result
    else:
        assert False


# calculate summarized gradient norm for all net
def get_net_gradients_norm(net: Net):
    result_norm = 0

    for layer in net.layers:
        grad_data = layer.get_gradient_data()

        if not (grad_data is None):
            w_grad_norm = la.norm(grad_data.weights_gradient)
            b_grad_norm = la.norm(grad_data.biases_gradient)

            result_norm += (w_grad_norm ** 2 + b_grad_norm ** 2)

    return math.sqrt(result_norm)
