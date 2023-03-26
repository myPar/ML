import numpy as np
from abc import ABC, abstractmethod


def print_matrix(matrix, offset):
    if len(matrix.shape) == 1:
        for i in matrix:
            print(offset + str(i), end='')
        print()

    elif len(matrix.shape) == 2:
        for row in range(len(matrix)):
            print(offset, end='')

            for column in range(len(matrix[row])):
                print(str(matrix[row][column]) + " ", end='')
            print()
    else:
        for i in range(len(matrix)):
            print(offset + "[" + str(i) + "]:")
            print_matrix(matrix[i], offset + "    ")


# uniform distribution from the specified interval
def uniform_init(interval: tuple, array_shape: tuple):
    assert len(interval) == 2
    max_val = interval[1]
    min_val = interval[0]

    delta = max_val - min_val
    assert delta > 0

    return min_val + np.random.rand(*array_shape) * delta


def default_init(array_shape: tuple):
    return uniform_init((-0.5, 0.5), array_shape)


class Optimizer(ABC):
    @abstractmethod
    def get_coefficient(self, *net_parameters):  # returns gradient coefficient and updates optimizer parameters
        pass


class Adam(Optimizer):
    def __init__(self):
        self.p1 = 0.9
        self.p2 = 0.99
        self.epsilon = 0.000001

        self.r_t = 0    # zero on the first step
        self.s_t = 0

    def get_coefficient(self, full_gradient_norm: float):
        pass