from abc import ABC, abstractmethod
import numpy as np
from enum import IntEnum

from TPNN_2023.lab2.activation_functions import *
from TPNN_2023.lab2.tools import print_matrix


class ConfigLevel(IntEnum):
    LOW = 1         # minimum layer info
    MEDIUM = 2      # add layer parameters
    HIGH = 3        # add gradient info


class Layer(ABC):
    @abstractmethod
    def calc_output(self, input_vector):
        pass

    @abstractmethod
    def learn_step(self, input_data):   # returns gradient array for input vector
        pass

    @abstractmethod
    def print_layer_config(self, config_level: int):
        pass


class Dense(Layer):
    def __init__(self, neuron_count, act_function):
        self.neuron_count = neuron_count
        self.input_vector = None
        self.activations = None
        self.z_array = None
        self.act_function = act_function

        self.weight_matrix = None
        self.biases = None

        self.z_grad_array = None
        self.weights_grad_matrix = None
        self.biases_grad_array = None

    def set_weight_matrix(self, matrix):
        assert matrix.shape[0] == self.neuron_count
        self.weight_matrix = matrix

    def calc_output(self, input_vector):
        # input vector - vector of output from the previous layer
        assert input_vector.shape == (self.weight_matrix.shape[1],)
        assert self.activations.shape == (self.neuron_count, )

        self.input_vector = input_vector
        self.activations = np.dot(self.weight_matrix, input_vector) + self.biases
        self.z_array = np.apply_along_axis(func1d=self.act_function, axis=0, arr=self.activations)

        return self.z_array

    # returns gradient array for input vector (z_array for the previous layer):
    def learn_step(self, prev_layer_z_array):
        assert self.z_grad_array is not None
        assert prev_layer_z_array.shape == (self.weight_matrix.shape[1],)

        # calc activation gradient array:
        yacobian = np.diag(np.apply_along_axis(func1d=get_der(self.act_function), axis=0, arr=self.activations))
        # no need to transpose, yacobian is a diagonal matrix
        act_grad_array = np.dot(yacobian, self.z_grad_array)

        # calc biases grad array
        biases_grad_array = act_grad_array

        # calc weighs grad matrix
        arg1 = np.array([act_grad_array]).transpose()
        arg2 = np.array([prev_layer_z_array])
        weights_grad_matrix = np.dot(arg1, arg2)

        assert weights_grad_matrix.shape == self.weight_matrix.shape
        assert biases_grad_array.shape == self.biases.shape

        # cache gradients for further changes of layer parameters
        self.weights_grad_matrix = weights_grad_matrix
        self.biases_grad_array = biases_grad_array

        # calc gradient array for the input vector
        z_prev_grad_array = np.dot(self.weight_matrix.transpose(), act_grad_array)

        return z_prev_grad_array

    def print_layer_config(self, config_level):
        print("Dense layer:")
        print("  neuron count=" + str(self.neuron_count))

        if config_level >= ConfigLevel.MEDIUM:
            print("weight matrix:")
            print_matrix(matrix=self.weight_matrix, offset="  ")
            print("biases:")
            print_matrix(self.biases, offset="  ")
        if config_level >= ConfigLevel.HIGH:
            print("weigh's gradient:")
            print_matrix(self.weights_grad_matrix, offset="  ")
            print("biases gradient:")
            print_matrix(self.biases_grad_array, offset="  ")


class Softmax(Layer):
    def __init__(self, neuron_count):
        self.neuron_count = neuron_count
        self.input_vector = None

    def calc_output(self, input_vector):
        assert input_vector.shape == (self.neuron_count,)
        self.input_vector = input_vector

        exp_arr = np.exp(input_vector)

        return exp_arr / np.sum(exp_arr)

    def learn_step(self, one_hot_enc_vector):
        assert np.sum(one_hot_enc_vector) == 1 and np.prod(one_hot_enc_vector) == 1

        return self.input_vector - one_hot_enc_vector

    def print_layer_config(self, config_level: int):
        pass


class Net:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer: Layer):
        self.layers.append(layer)

    def train_step(self, input_vector, target_vector):
        pass

    def train(self, input_train_data, target_train_data, epoch_count: int):
        pass

    def calc_output(self, input_vector):
        pass
