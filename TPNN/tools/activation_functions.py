import numpy as np

# activation functions
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
