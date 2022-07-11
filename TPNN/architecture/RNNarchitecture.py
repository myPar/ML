import numpy as np
from numpy import random
from TPNN.architecture.functions import *


class LSTM(object):
    def __init__(self, input_shape, output_size, timesteps):
        self.name = 'lstm'
        input_dim = len(input_shape)

        assert 1 <= input_dim <= 2 and self.name + " - invalid input shape dim"
        self.input_shape = input_shape

        # init base fields:
        self.timesteps = timesteps
        self.output_size = output_size

        x_size = input_shape[0] if input_dim == 1 else input_shape[0] * input_shape[1]
        h_size = output_size

        self.input_sequence_shape = (timesteps,) + x_size
        self.x_sequence = None
        self.output_sequence = None

        # init net parameters:
        self.U_i = random.rand(h_size, x_size)
        self.U_f = random.rand(h_size, x_size)
        self.U_p = random.rand(h_size, x_size)
        self.U_o = random.rand(h_size, x_size)

        self.W_i = random.rand(h_size, h_size)
        self.W_f = random.rand(h_size, h_size)
        self.W_p = random.rand(h_size, h_size)
        self.W_o = random.rand(h_size, h_size)

        self.b_i = random.rand(h_size, 1)
        self.b_f = random.rand(h_size, 1)
        self.b_p = random.rand(h_size, 1)
        self.b_o = random.rand(h_size, 1)

        # init lambda set (i,f,p,o) for each timestep:
        self.i = np.zeros((timesteps, h_size, 1))
        self.f = np.zeros((timesteps, h_size, 1))
        self.p = np.zeros((timesteps, h_size, 1))
        self.o = np.zeros((timesteps, h_size, 1))

        # init h and c value's arrays:
        self.h = np.zeros((timesteps, h_size, 1))
        self.c = np.zeros((timesteps, h_size, 1))

        # init gradients (h, c, lambda set, net parameters derivatives):
        self.i_der_array = np.zeros((timesteps, h_size, 1))
        self.f_der_array = np.zeros((timesteps, h_size, 1))
        self.p_der_array = np.zeros((timesteps, h_size, 1))
        self.o_der_array = np.zeros((timesteps, h_size, 1))

        self.h_der_array = np.zeros((timesteps, h_size, 1))
        self.c_der_array = np.zeros((timesteps, h_size, 1))

        self.W_i_der_array = np.zeros((h_size, h_size))
        self.W_f_der_array = np.zeros((h_size, h_size))
        self.W_p_der_array = np.zeros((h_size, h_size))
        self.W_o_der_array = np.zeros((h_size, h_size))

        self.U_i_der_array = np.zeros((h_size, x_size))
        self.U_f_der_array = np.zeros((h_size, x_size))
        self.U_p_der_array = np.zeros((h_size, x_size))
        self.U_o_der_array = np.zeros((h_size, x_size))

        self.b_i_der_array = np.zeros((h_size, 1))
        self.b_f_der_array = np.zeros((h_size, 1))
        self.b_p_der_array = np.zeros((h_size, 1))
        self.b_o_der_array = np.zeros((h_size, 1))

    def get_node(self, U_matrix, W_matrix, biase, timestep, function, first_block: bool):
        x_vector = self.x_sequence[timestep]

        if first_block:
            arg = np.matmul(U_matrix, x_vector) + biase
        else:
            arg = np.matmul(U_matrix, x_vector) + np.matmul(W_matrix, self.h[timestep - 1]) + biase

        return np.apply_along_axis(function, 0, arg)

    def get_output(self, input_sequence):
        assert input_sequence.shape == self.input_sequence_shape

        self.output_sequence = np.zeros((self.timesteps, self.output_size))

        # reshape input sequence
        self.x_sequence = input_sequence.reshape(self.input_sequence_shape)

        # init first block:
        self.f[0] = self.get_node(self.U_f, None, self.b_f, 0, sigmoid, True)
        self.o[0] = self.get_node(self.U_o, None, self.b_o, 0, sigmoid, True)
        self.i[0] = self.get_node(self.U_i, None, self.b_i, 0, sigmoid, True)
        self.p[0] = self.get_node(self.U_p, None, self.b_p, 0, th, True)

        self.c[0] = self.f[0] + self.i[0] * self.p[0]
        self.h[0] = self.o[0] * np.apply_along_axis(th, 0, self.c[0])

        self.output_sequence[0] = np.apply_along_axis()

        for t in range(1, self.timesteps):
            self.f[t] = self.get_node(self.U_f, self.W_f, self.b_f, t, sigmoid, True)
            self.o[t] = self.get_node(self.U_o, self.W_o, self.b_o, t, sigmoid, True)
            self.i[t] = self.get_node(self.U_i, self.W_i, self.b_i, t, sigmoid, True)
            self.p[t] = self.get_node(self.U_p, self.W_p, self.b_p, t, th, True)

            self.c[t] = self.c[t - 1] * self.f[t] + self.i[t] * self.p[t]
            self.h[t] = self.o[t] * np.apply_along_axis(th, 0, self.c[t])
