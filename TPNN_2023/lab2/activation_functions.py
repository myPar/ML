import numpy as np


def ReLU(x):
    result = np.zeros((len(x),))

    for i in range(len(x)):
        if x[i] >= 0:
            result[i] = x[i]
        else:
            result[i] = 0

    return result


def ident(x):
    return x


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def th(x):
    return np.tanh(x)


def ELU(x):
    alpha = 1
    result = np.zeros((len(x),))

    for i in range(len(x)):
        if x[i] >= 0:
            result[i] = x[i]
        else:
            result[i] = alpha * (np.exp(x[i]) - 1)

    return result


# derivatives of activation functions
def ident_der(x):
    return np.ones((len(x),))


def sigmoid_der(x):
    sigma = sigmoid(x)

    return sigma * (1 - sigma)


def th_der(x):
    return 1 - th(x) ** 2


def ELU_der(x):  # operates over array of data
    alpha = 1
    result = np.zeros((len(x),))

    for i in range(len(x)):
        if x[i] >= 0:
            result[i] = 1
        else:
            result[i] = alpha * np.exp(x[i])

    return result


def ReLU_der(x):
    result = np.zeros((len(x),))

    for i in range(len(x)):
        if x[i] >= 0:
            result[i] = 1
        else:
            result[i] = 0

    return result


def get_der(function_ident):
    if function_ident == sigmoid:
        return sigmoid_der
    elif function_ident == ident:
        return ident_der
    elif function_ident == th:
        return th_der
    elif function_ident == ELU:
        return ELU_der
    elif function_ident == ReLU:
        return ReLU_der
    else:
        assert False
