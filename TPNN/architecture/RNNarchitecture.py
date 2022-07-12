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

        # cacheable parameters:
        self.c1_h_y = None
        self.diag_o = None
        self.diag_th_c1 = None

        self.diag_sigma_f = None
        self.diag_th_p = None
        self.diag_sigma_i = None
        self.diag_sigma_o = None

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

        self.output_sequence[0] = softmax(self.h[0])

        for t in range(1, self.timesteps):
            self.f[t] = self.get_node(self.U_f, self.W_f, self.b_f, t, sigmoid, True)
            self.o[t] = self.get_node(self.U_o, self.W_o, self.b_o, t, sigmoid, True)
            self.i[t] = self.get_node(self.U_i, self.W_i, self.b_i, t, sigmoid, True)
            self.p[t] = self.get_node(self.U_p, self.W_p, self.b_p, t, th, True)

            self.c[t] = self.c[t - 1] * self.f[t] + self.i[t] * self.p[t]
            self.h[t] = self.o[t] * np.apply_along_axis(th, 0, self.c[t])

            self.output_sequence[t] = softmax(self.h[t])

        return self.output_sequence

    # calc c derivative methods:
    def calc_h1_c_y(self, timestep):
        diag_o = np.diagflat(self.o[timestep])
        diag_th_c1 = np.diagflat(1 - np.apply_along_axis(th, 0, self.c[timestep + 1]) ** 2)
        diag_f = np.diagflat(self.f[timestep])

        return np.matmul(np.matmul(diag_o, diag_th_c1), diag_f)

    def calc_h_c_y(self, timestep):
        diag_o = np.diagflat(self.o[timestep])
        diag_th_c1 = np.diagflat(1 - np.apply_along_axis(th, 1, self.c[timestep + 1]) ** 2)

        return np.matmul(diag_o, diag_th_c1)

    # note: c1_c_y - dc^t+1/dc^t yacobian
    def calc_c_der_item(self, timestep, expected_output_sequence):
        if timestep == self.timesteps - 1:
            diag_o = np.diagflat(self.o[timestep])
            diag_th_c = np.diagflat(1 - np.apply_along_axis(th, 0, self.c[timestep]) ** 2)
            h_gradient = self.h_der_array[timestep]

            self.c_der_array[timestep] = np.matmul(np.matmul(diag_o, diag_th_c), h_gradient)
        else:
            h1_c_y =  self.calc_h1_c_y(timestep)
            h_c_y = self.calc_h_c_y(timestep)
            c1_c_y = np.diagflat(self.f[timestep])

            self.c_der_array[timestep] = np.matmul(c1_c_y.T, self.c_der_array[timestep + 1]) + \
                                         np.matmul(h1_c_y.T, self.h_der_array[timestep + 1]) + \
                                         np.matmul(h_c_y.T, self.output_sequence[timestep] - expected_output_sequence[timestep])

    def calc_h1_h_y(self, timestep, expected_output_sequence):
        diag_o = np.diagflat(self.o[timestep])
        diag_th_c1 = np.diagflat(1 - np.apply_along_axis(th, 0, self.c[timestep + 1]) ** 2)

    def calc_h_der_item(self, timestep, expected_output_sequnce):
        if timestep == self.timesteps - 1:
            return self.output_sequence[timestep] - expected_output_sequnce[timestep]
        else:
            h1_h_y = self.calc_h1_h_y(timestep, expected_output_sequnce)

    def calc_derivatives(self, expected_output_sequence):
        assert expected_output_sequence.shape == self.output_sequence.shape
        last_idx = self.timesteps - 1

        # 1. calc h and c gradients
        # 2. calc node gradients
        # 3. calc U,W,b gradients

        # calc last timestep h and c gradients:
        for t in range(self.timesteps - 2, -1, -1):
            if t < self.timesteps - 1:
                # calc cacheable parameters:
                self.diag_th_c1 = np.diagflat(1 - np.apply_along_axis(th, 0, self.c[t + 1]) ** 2)
                self.diag_o = np.diagflat(self.o[t])
                self.c1_h_y = self.calc_c1_h_y(t, expected_output_sequence)

                self.diag_sigma_f = np.diagflat((1 - self.f[t]) * self.f[t])
                self.diag_th_p = np.diagflat(1 - self.p[t] ** 2)
                self.diag_sigma_i = np.deagflat((1 - self.i[t]) * self.i[t])
                self.diag_sigma_o = np.deagflat((1 - self.o[t]) * self.o[t])

    def calc_c1_h_y(self, timestep, expected_output_sequence):
        def calc_f_h_y():
            return np.matmul(self.diag_sigma_f, self.W_f)

