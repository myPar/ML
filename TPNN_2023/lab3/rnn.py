"""
many-to-many rnn with batch size equals to timestamps number for solving regression task
(L = 1/2 * sum_i=1^n (y_i - o_i)) - loss calculation
1. forward prop: need to cache output and input vectors, dst output vectors, args of activation function in calculation of h^t
2. back prop:
    2.1. calc o^t gradients
    2.2. calc h^t gradients
    2.3. calc net parameters gradients: W, U, V, b^h, b^o

all 1-d vectors are vector-column!
"""
import math

import numpy as np

from TPNN_2023.lab2.Perceptron import Layer, ConfigLevel
from TPNN_2023.lab2.activation_functions import get_der
from TPNN_2023.lab2.tools import default_init, print_matrix, Optimizer


class RNNlayer(Layer):
    def __init__(self, timestamps: int, neuron_count: int, output_vector_shape: tuple, activation,
                 input_vector_shape: tuple):
        assert timestamps > 0
        assert neuron_count > 0
        assert len(output_vector_shape) == 2 and output_vector_shape[0] > 0 and output_vector_shape[1] == 1
        assert len(input_vector_shape) == 2 and input_vector_shape[0] > 0 and input_vector_shape[1] == 1

        # init parameters
        self.timestamps = timestamps
        self.neuron_count = neuron_count
        self.output_vector_shape = output_vector_shape
        self.input_vector_shape = input_vector_shape
        self.activation = activation
        # net parameters (changed by backprop)
        self.W = default_init(array_shape=(neuron_count, neuron_count))
        self.U = default_init(array_shape=(neuron_count, self.input_vector_shape[0]))
        self.V = default_init(array_shape=(output_vector_shape[0], neuron_count))
        self.b_h = default_init(array_shape=(neuron_count, 1))              # vector-columns
        self.b_o = default_init(array_shape=(output_vector_shape[0], 1))    #
        # caching parameters
        self.output_vectors = np.zeros((timestamps, output_vector_shape[0], 1))
        self.input_vectors = None
        self.dst_vectors = None
        self.h_vectors = np.zeros((timestamps, neuron_count, 1))
        self.h_args = np.zeros((timestamps, neuron_count, 1))
        # gradients (of net parameters)
        self.W_grad = np.zeros(self.W.shape)
        self.U_grad = np.zeros(self.U.shape)
        self.V_grad = np.zeros(self.V.shape)
        self.b_h_grad = np.zeros(self.b_h.shape)
        self.b_o_grad = np.zeros(self.b_o.shape)
        # gradients (for calculation)
        self.o_grad_array = np.zeros((timestamps, self.output_vector_shape[0], 1))
        self.h_grad_array = np.zeros((timestamps, self.neuron_count, 1))

    def calc_output(self, input_tensor):
        # input_tensor.shape == (timestamps, input_vector.shape[0])
        assert len(input_tensor.shape) == 3 and input_tensor.shape[0] == self.timestamps \
               and input_tensor.shape[1] == self.input_vector_shape[0] and input_tensor.shape[2] == 1
        # cache input vector
        self.input_vectors = input_tensor

        # init first block
        x_0 = input_tensor[0]
        arg = np.matmul(self.U, x_0) + self.b_h
        self.h_args[0] = arg
        self.h_vectors[0] = np.apply_along_axis(func1d=self.activation, axis=0, arr=arg)
        self.output_vectors[0] = np.matmul(self.V, self.h_vectors[0]) + self.b_o

        # init remaining blocks
        for timestamp in range(1, self.timestamps, 1):
            x_t = input_tensor[timestamp]
            arg = np.matmul(self.U, x_t) + np.matmul(self.W, self.h_vectors[timestamp - 1]) + self.b_h
            self.h_args[timestamp] = arg
            self.h_vectors[timestamp] = np.apply_along_axis(func1d=self.activation, axis=0, arr=arg)
            self.output_vectors[timestamp] = np.matmul(self.V, self.h_vectors[timestamp]) + self.b_o

        return self.output_vectors

    def learn_step(self, arg_vector, is_last_layer: bool):  # returns x_grad array
        assert arg_vector.shape == (self.timestamps, self.output_vector_shape[0], 1)

        # caching calculation arg
        diag_der_array = np.zeros((self.timestamps, self.neuron_count, self.neuron_count))

        if is_last_layer:
            # calc o^t gradients
            for timestamp in range(self.timestamps):
                # arg vector - is dst_vector
                self.o_grad_array[timestamp] = self.output_vectors[timestamp] - arg_vector[timestamp]
        else:
            # arg vector - is input_grad_array vector of the next layer
            self.o_grad_array = arg_vector

        # calc h^t gradient for last timestamp
        tau = self.timestamps - 1
        self.h_grad_array[tau] = np.matmul(self.V.T, self.o_grad_array[tau])
        diag_der_array[tau] = np.diag(np.apply_along_axis(func1d=get_der(self.activation),
                                                          axis=0, arr=self.h_args[tau]).reshape(self.neuron_count, ))
        # calc h^t gradients for remaining timestamps
        for timestamp in range(self.timestamps - 2, -1, -1):
            diag_der_array[timestamp] = np.diag(np.apply_along_axis(func1d=get_der(self.activation), axis=0,
                                                                    arr=self.h_args[timestamp]).reshape(self.neuron_count, ))
            self.h_grad_array[timestamp] = np.matmul(self.V.T, self.o_grad_array[timestamp]) + \
                                                     np.matmul(np.matmul(self.W.T, diag_der_array[timestamp]),
                                                     self.h_grad_array[timestamp + 1])
        # calc net parameters gradients:
        b_h_grad = np.zeros(self.b_h.shape)
        b_o_grad = np.zeros(self.b_o.shape)
        w_grad = np.zeros(self.W.shape)
        u_grad = np.zeros(self.U.shape)
        v_grad = np.zeros(self.V.shape)

        result = np.zeros((self.timestamps, self.input_vector_shape[0], 1))

        for timestamp in range(self.timestamps):
            b_h_grad += np.matmul(diag_der_array[timestamp], self.h_grad_array[timestamp])
            b_o_grad += self.o_grad_array[timestamp]

            if timestamp == 0:
                h_prev = np.zeros((self.neuron_count, 1))
            else:
                h_prev = self.h_vectors[timestamp - 1]

            w_grad += np.matmul(np.matmul(diag_der_array[timestamp], self.h_grad_array[timestamp]),
                                h_prev.T)
            x_t_row = self.input_vectors[timestamp].reshape((1, self.input_vector_shape[0]))
            u_grad += np.matmul(np.matmul(diag_der_array[timestamp], self.h_grad_array[timestamp]),
                                x_t_row)
            v_grad += np.matmul(self.o_grad_array[timestamp], self.h_vectors[timestamp].T)
            result[timestamp] = np.matmul(np.matmul(diag_der_array[timestamp], self.U).T, self.h_grad_array[timestamp])

        # set new values to net parameter gradients
        self.b_h_grad = b_h_grad
        self.b_o_grad = b_o_grad
        self.W_grad = w_grad
        self.U_grad = u_grad
        self.V_grad = v_grad

        return result

    def print_layer_config(self, config_level: int, offset: str):
        print(offset + "RNN layer:")
        print(offset + "  neuron count=" + str(self.neuron_count))

        if config_level >= ConfigLevel.MEDIUM:
            print(offset + "matrices:")
            print(2 * offset + "W matrix:")
            print_matrix(matrix=self.W, offset=2 * offset + "  ")
            print(2 * offset + "U matrix:")
            print_matrix(matrix=self.U, offset=2 * offset + "  ")
            print(2 * offset + "V matrix:")
            print_matrix(matrix=self.V, offset=2 * offset + "  ")

            print(offset + "biases:")
            print(2 * offset + "b_h:")
            print_matrix(self.b_h, offset=2 * offset + "  ")
            print(2 * offset + "b_o:")
            print_matrix(self.b_o, offset=2 * offset + "  ")

        if config_level >= ConfigLevel.HIGH:
            print(offset + "weight's gradients:")
            print(2 * offset + "W gradient:")
            print_matrix(matrix=self.W_grad, offset=2 * offset + "  ")
            print(2 * offset + "U gradient:")
            print_matrix(matrix=self.U_grad, offset=2 * offset + "  ")
            print(2 * offset + "V gradient:")
            print_matrix(matrix=self.V_grad, offset=2 * offset + "  ")

            print(offset + "bias's gradients:")
            print(2 * offset + "b_h gradient:")
            print_matrix(self.b_h_grad, offset=2 * offset + "  ")
            print(2 * offset + "b_o gradient:")
            print_matrix(self.b_o_grad, offset=2 * offset + "  ")

    def get_layer_type(self):
        pass

    def init_layer(self, init_strategy):
        pass

    def set_input_shape(self, input_shape):
        pass

    def update_parameters(self, update_coefficient):
        assert self.W_grad is not None
        assert self.U_grad is not None
        assert self.V_grad is not None
        assert self.b_h_grad is not None
        assert self.b_o_grad is not None

        l = lambda x: x * update_coefficient
        self.W -= np.apply_along_axis(func1d=l, axis=0, arr=self.W_grad)
        self.U -= np.apply_along_axis(func1d=l, axis=0, arr=self.U_grad)
        self.V -= np.apply_along_axis(func1d=l, axis=0, arr=self.V_grad)
        self.b_h -= np.apply_along_axis(func1d=l, axis=0, arr=self.b_h_grad)
        self.b_o -= np.apply_along_axis(func1d=l, axis=0, arr=self.b_o_grad)

    def calc_full_gradient_norm(self):
        assert self.W_grad is not None
        assert self.U_grad is not None
        assert self.V_grad is not None
        assert self.b_h_grad is not None
        assert self.b_o_grad is not None

        concatenated_gradients_array = np.concatenate((self.W_grad.flatten(), self.U_grad.flatten(),
                                                       self.V_grad.flatten(), self.b_h_grad.flatten(),
                                                       self.b_o_grad.flatten()), axis=0)
        return np.linalg.norm(concatenated_gradients_array)

    def reset_timestamps(self, timestamps):
        assert timestamps > 0
        self.timestamps = timestamps
        self.output_vectors = np.zeros((timestamps, self.output_vector_shape[0], 1))
        self.h_vectors = np.zeros((timestamps, self.neuron_count, 1))
        self.h_args = np.zeros((timestamps, self.neuron_count, 1))
        self.o_grad_array = np.zeros((timestamps, self.output_vector_shape[0], 1))
        self.h_grad_array = np.zeros((timestamps, self.neuron_count, 1))


class RNNnet:
    def __init__(self):
        self.layers = []
        self.layers_count = 0
        self.loss_list = []         # epoch's losses list
        self.loss_function = None

    def add_layer(self, rnn_layer: RNNlayer):
        self.layers.append(rnn_layer)
        self.layers_count += 1

    def calc_output(self, input_tensor):
        result = input_tensor

        for layer in self.layers:
            result = layer.calc_output(result)

        return result

    def train_step(self, input_tensor, dst_tensor, optimizer: Optimizer, lr: float) -> float:
        # forward prop
        predicted = self.calc_output(input_tensor)
        loss = self.loss_function(predicted, dst_tensor)

        input_grad_array = dst_tensor

        # back prop (gradients calculation):
        for layer_idx in range(self.layers_count - 1, -1, -1):
            is_last_layer = layer_idx == self.layers_count - 1

            input_grad_array = self.layers[layer_idx].learn_step(input_grad_array, is_last_layer)

        # back prop (updating parameters):
        full_gradient_norm = self.calc_net_gradient_norm()
        for layer in self.layers:
            layer.update_parameters(lr * optimizer.get_coefficient(full_gradient_norm))

        return loss

    def train(self, input_train_data, target_train_data, optimizer: Optimizer, loss, epoch_count: int, lr: float):
        samples_count = len(input_train_data)
        assert samples_count == len(target_train_data)
        self.loss_function = loss

        for epoch in range(epoch_count):
            epoch_losses = []

            for i in range(samples_count):
                sample = input_train_data[i]
                target_sample = target_train_data[i]

                sample_loss = self.train_step(input_tensor=sample, dst_tensor=target_sample, optimizer=optimizer, lr=lr)
                epoch_losses.append(sample_loss)
            epoch_losses = np.array(epoch_losses)
            cost = np.average(epoch_losses)
            print("loss=" + str(cost))

            self.loss_list.append(cost)

    def calc_net_gradient_norm(self):
        result = 0

        for layer in self.layers:
            result += layer.calc_full_gradient_norm() ** 2  # frobenius norm is used
        assert result >= 0

        return math.sqrt(result)

    def print_net_config(self, level: ConfigLevel):
        assert self.layers_count == len(self.layers)
        print("layers count=" + str(self.layers_count))
        print("layers:")

        for i in range(len(self.layers)):
            cur_layer = self.layers[i]
            print("[" + str(i) + "]")
            cur_layer.print_layer_config(config_level=level, offset="  ")

    def test(self, input_test_data, target_test_data, loss) -> float:   # returns average loss of predictions
        samples_count = len(input_test_data)
        assert samples_count == len(target_test_data)
        losses = []

        for i in range(samples_count):
            sample = input_test_data[i]
            target_sample = target_test_data[i]

            output = self.calc_output(sample)
            sample_loss = loss(output, target_sample)
            losses.append(sample_loss)

        return np.average(np.array(losses))

    def reset_layers_timestamps(self, timestamps):
        for layer in self.layers:
            layer.reset_timestamps(timestamps)
