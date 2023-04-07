import math

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


def test_init(array_shape: tuple):
    return np.ones(array_shape)


class Optimizer(ABC):
    @abstractmethod
    def get_coefficient(self, *net_parameters):  # returns gradient coefficient and updates optimizer parameters
        pass


class Adam(Optimizer):
    def __init__(self):
        self.betta1 = 0.9
        self.betta2 = 0.99
        self.epsilon = 0.000001

        self.m_t = 0    # zero on the first step
        self.v_t = 0
        self.step = 1

    def get_coefficient(self, full_gradient_norm: float):
        assert full_gradient_norm >= 0

        next_m_t = self.m_t * self.betta1 + (1 - self.betta1) * full_gradient_norm
        next_v_t = self.v_t * self.betta2 + (1 - self.betta2) * (full_gradient_norm ** 2)

        arg1 = self.m_t / (1 - self.betta1)
        arg2 = self.v_t / (1 - self.betta2)

        coefficient = self.step * arg1 / (math.sqrt(arg2) + self.epsilon)

        self.step += 1
        self.m_t = next_m_t
        self.v_t = next_v_t

        return coefficient
