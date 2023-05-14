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
import numpy as np

from TPNN_2023.lab2.Perceptron import Layer
from TPNN_2023.lab2.activation_functions import get_der
from TPNN_2023.lab2.tools import default_init


class RNNlayer(Layer):
    def __init__(self, timestamps: int, neuron_count: int, output_vector_shape: tuple, activation,
                 input_vector_shape: tuple):
        assert timestamps > 0
        assert neuron_count > 0
        assert len(output_vector_shape) == 1 and output_vector_shape[0] > 0
        assert len(input_vector_shape) == 1 and input_vector_shape[0] > 0

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
        self.b_h = default_init(array_shape=(neuron_count, 1))
        self.b_o = default_init(array_shape=(output_vector_shape[0], 1))
        # caching parameters
        self.output_vectors = None
        self.input_vectors = None
        self.dst_vectors = None
        self.h_vectors = np.zeros((timestamps, neuron_count))
        self.h_args = None
        # gradients (of net parameters)
        self.W_grad = None
        self.U_grad = None
        self.V_grad = None
        self.b_h_grad = None
        self.b_o_grad = None
        # gradients (for calculation)
        self.o_grad_array = np.zeros((timestamps, self.output_vector_shape[0], 1))
        self.h_grad_array = np.zeros((timestamps, self.neuron_count, 1))

    def calc_output(self, input_tensor):
        # input_tensor.shape == (timestamps, input_vector.shape[0])
        assert len(input_tensor.shape) == 2 and input_tensor.shape[0] == self.timestamps \
               and input_tensor.shape[1] == self.input_vector_shape[0]
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
            self.h_vectors[timestamp] = np.apply_along_axis(func1d=self.activation, axis=0, arg=arg)
            self.output_vectors[timestamp] = np.matmul(self.V, self.h_vectors[timestamp]) + self.b_o

    def learn_step(self, dst_vectors):
        assert dst_vectors.shape == (self.timestamps, self.output_vector_shape[0])

        # caching calculation arg
        diag_der_array = np.zeros((self.timestamps, self.neuron_count, self.neuron_count))

        # calc o^t gradients
        for timestamp in range(self.timestamps):
            self.o_grad_array[timestamp] = self.output_vectors[timestamp] - dst_vectors[timestamp]

        # calc h^t gradients for last timestamp
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
        b_h_grad = np.zeros((self.neuron_count, 1))
        b_o_grad = np.zeros((self.output_vector_shape[0], 1))
        w_grad = np.zeros(self.W.shape)
        u_grad = np.zeros(self.U.shape)
        v_grad = np.zeros(self.V.shape)

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
        # set new values to net parameter gradients
        self.b_h_grad = b_h_grad
        self.b_o_grad = b_o_grad
        self.W_grad = w_grad
        self.U_grad = u_grad
        self.V_grad = v_grad

    def print_layer_config(self, config_level: int, offset: str):
        pass

    def get_layer_type(self):
        pass

    def init_layer(self, init_strategy):
        pass

    def set_input_shape(self, input_shape):
        pass

    def update_parameters(self, *update_args):
        pass

    def calc_full_gradient_norm(self):
        pass
