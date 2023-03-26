from abc import ABC, abstractmethod
import numpy as np
from enum import IntEnum, StrEnum

from TPNN_2023.lab2.activation_functions import *
from TPNN_2023.lab2.tools import print_matrix, Optimizer


class ConfigLevel(IntEnum):
    LOW = 1         # minimum layer info
    MEDIUM = 2      # add layer parameters
    HIGH = 3        # add gradient info


class LayerType(StrEnum):
    DENSE = "Dense"
    SOFTMAX = "Softmax"


class Layer(ABC):
    @abstractmethod
    def calc_output(self, input_vector):
        pass

    @abstractmethod
    def learn_step(self, *input_data):   # returns gradient array for input vector
        pass

    @abstractmethod
    def print_layer_config(self, config_level: int):
        pass

    @abstractmethod
    def get_layer_type(self):
        pass

    @abstractmethod
    def init_layer(self, init_strategy):
        pass

    @abstractmethod
    def set_input_shape(self, input_shape):
        pass


class Dense(Layer):
    def __init__(self, neuron_count, act_function):
        self.neuron_count = neuron_count
        self.input_shape = None
        self.act_function = act_function

        self.input_vector = None
        self.activations = None
        self.z_array = None

        self.weight_matrix = None
        self.biases = None

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
    def learn_step(self, z_grad_array):
        assert z_grad_array is not None

        # calc activation gradient array:
        yacobian = np.diag(np.apply_along_axis(func1d=get_der(self.act_function), axis=0, arr=self.activations))
        # no need to transpose, yacobian is a diagonal matrix
        act_grad_array = np.dot(yacobian, z_grad_array)

        # calc biases grad array
        biases_grad_array = act_grad_array

        # calc weighs grad matrix
        arg1 = np.array([act_grad_array]).transpose()
        arg2 = np.array([self.input_vector])
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

    def get_layer_type(self):
        return LayerType.DENSE

    def set_input_shape(self, shape):
        assert len(shape) == 1
        self.input_shape = shape

    def init_layer(self, init_strategy):
        assert self.input_shape is not None
        # init matrix based on in shape and neurons count
        weight_matrix_shape = (self.neuron_count, self.input_shape[0])
        biases_shape = (self.neuron_count,)

        self.weight_matrix = init_strategy(weight_matrix_shape)
        self.biases = init_strategy(biases_shape)


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

    def get_layer_type(self):
        return LayerType.SOFTMAX

    def init_layer(self, init_strategy):
        pass

    def set_input_shape(self, input_shape):
        pass


class Net:
    def __init__(self):
        self.layers = []
        self.layers_count = 0

    def add_layer(self, layer: Layer):
        self.layers.append(layer)
        self.layers_count += 1

    def train_step(self, input_vector, target_vector):  # return loss value for the sample
        in_v = input_vector

        # forward propagation
        for layer in self.layers:
            in_v = layer.calc_output(in_v)

        last_idx = self.layers_count - 1
        z_grad_array = None

        for i in range(last_idx, -1, 0):
            cur_layer = self.layers[i]

            layer_type = cur_layer.get_layer_type()
            if i == last_idx and layer_type == LayerType.SOFTMAX:   # last layer is softmax layer
                z_grad_array = cur_layer.learn_step(target_vector)
            elif layer_type == LayerType.DENSE:
                z_grad_array = cur_layer.learn_step(z_grad_array)
            else:
                assert False

    def train(self, input_train_data, target_train_data, optimizer: Optimizer, learning_rate: float, epoch_count: int):
        samples_count = len(input_train_data)
        assert samples_count == len(target_train_data)

        for epoch in range(epoch_count):
            for i in range(samples_count):
                sample = input_train_data[i]
                target_sample = target_train_data[i]

                self.train_step(sample, target_sample)

    def calc_output(self, input_vector):
        in_v = input_vector

        for layer in self.layers:
            in_v = layer.calc_output(in_v)

        return in_v

    def init_net(self, init_strategy, input_shape):
        assert len(input_shape) == 1

        for i in range(len(self.layers)):
            cur_layer = self.layers[i]

            if i == 0:
                assert cur_layer.get_layer_type == LayerType.DENSE
                cur_layer.set_input_shape(input_shape)
            else:
                prev_layer = self.layers[i - 1]
                cur_input_shape = (prev_layer.neuron_count,)

                cur_layer.set_input_shape(cur_input_shape)
            cur_layer.init_layer(init_strategy)
