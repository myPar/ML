import numpy as np
from numpy import random
from TPNN.architecture.functions import *

"""""
timestep - number of current recurrent block (from 0 to timesteps - 1)

let: timestep = t
then: h_t = h[t-1] , c_t = c[t-1] and h_t+1 = h[t], c_t+1 = c[t]
last timestep = self.timesteps - 1
so h values are: h_1, h_2, ..., h_timestep
"""""


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
        self.diag_i = None
        self.diag_f = None
        self.diag_p = None

        self.diag_c = None

        self.diag_sigma_f_der = None
        self.diag_th_p_der = None
        self.diag_sigma_i_der = None
        self.diag_sigma_o_der = None

        self.diag_th_c1_der = None
        self.h1_h_y = None
        self.c1_c_y = None
        self.h1_c_y = None
        self.h_c_y = None

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
            self.f[t] = self.get_node(self.U_f, self.W_f, self.b_f, t, sigmoid, False)
            self.o[t] = self.get_node(self.U_o, self.W_o, self.b_o, t, sigmoid, False)
            self.i[t] = self.get_node(self.U_i, self.W_i, self.b_i, t, sigmoid, False)
            self.p[t] = self.get_node(self.U_p, self.W_p, self.b_p, t, th, False)

            self.c[t] = self.c[t - 1] * self.f[t] + self.i[t] * self.p[t]
            self.h[t] = self.o[t] * np.apply_along_axis(th, 0, self.c[t])

            self.output_sequence[t] = softmax(self.h[t])

        return self.output_sequence

    def calc_derivatives(self, expected_output_sequence):
        assert expected_output_sequence.shape == self.output_sequence.shape

        # reset derivatives arrays:
        self.W_f_der_array *= 0
        self.W_o_der_array *= 0
        self.W_i_der_array *= 0
        self.W_p_der_array *= 0

        self.U_f_der_array *= 0
        self.U_o_der_array *= 0
        self.U_i_der_array *= 0
        self.U_p_der_array *= 0

        self.b_f_der_array *= 0
        self.b_o_der_array *= 0
        self.b_i_der_array *= 0
        self.b_p_der_array *= 0

        # 1. calc h and c gradients
        # 2. calc node gradients
        # 3. calc U,W,b gradients
        timesteps = self.timesteps

        # calc h1_t_end, c1_t_end gradients (t_end - last timestep)
        c1_t_end = self.c[timesteps - 1]

        h1_t_end_grad = self.h_der_array[timesteps - 1] = self.output_sequence[timesteps - 1] - expected_output_sequence[timesteps]
        self.c_der_array[timesteps - 1] = np.matmul(np.matmul(np.diagflat(self.o[timesteps - 1]),
                                                    np.diagflat(1 - np.apply_along_axis(th, 0, c1_t_end) ** 2)).T, h1_t_end_grad)

        # calc last timestep h and c gradients:

        # calc cacheable parameters:
        for t in range(self.timesteps - 1, -1, -1):
            c1_t = self.c[t]
            self.diag_th_c1_der = np.diagflat(1 - np.apply_along_axis(th, 0, c1_t) ** 2)

            self.diag_o = np.diagflat(self.o[t])
            self.diag_f = np.diagflat(self.f[t])
            self.diag_i = np.diagflat(self.i[t])
            self.diag_p = np.diagflat(self.p[t])

            self.diag_c = np.diagflat(self.c[t - 1])

            self.diag_sigma_f_der = np.diagflat((1 - self.f[t]) * self.f[t])
            self.diag_th_p_der = np.diagflat(1 - self.p[t] ** 2)
            self.diag_sigma_i_der = np.deagflat((1 - self.i[t]) * self.i[t])
            self.diag_sigma_o_der = np.deagflat((1 - self.o[t]) * self.o[t])

            c1_h_y = self.calc_c1_h_y(t)
            h1_h_y = self.calc_h1_h_y(t)
            c1_c_y = self.diag_f
            h1_c_y = self.calc_h1_c_y()
            h_c_y = self.calc_h_c_y(t)

            # calc h and c gradients:
            if t > 0:
                h_t_grad = self.output_sequence[t - 1] - expected_output_sequence[t - 1]
                self.h_der_array[t - 1] = np.matmul(h1_h_y.T, self.h_der_array[t]) + \
                                          np.matmul(c1_h_y.T, self.c_der_array[t]) + \
                                          h_t_grad
                self.c_der_array[t - 1] = np.matmul(c1_c_y.T, self.c_der_array[t]) + \
                                          np.matmul(h1_c_y.T, self.h_der_array[t]) + \
                                          np.matmul(h_c_y.T, h_t_grad)

            # calc i,c,o,f gradients:
            h1_i_y = self.calc_h1_i_y()
            c1_i_y = self.diag_p

            h1_f_y = self.calc_h1_f_y()
            c1_f_y = self.diag_c

            h1_o_y = self.calc_h1_o_y(t)

            h1_p_y = self.calc_h1_p_y()
            c1_p_y = self.diag_i

            self.i_der_array[t] = np.matmul(h1_i_y.T, self.h_der_array[t]) + \
                                  np.matmul(c1_i_y.T, self.c_der_array[t])
            self.f_der_array[t] = np.matmul(h1_f_y.T, self.h_der_array[t]) + \
                                  np.matmul(c1_f_y.T, self.c_der_array[t])
            self.o_der_array[t] = np.matmul(h1_o_y.T, self.h_der_array[t])
            self.p_der_array[t] = np.matmul(h1_p_y.T, self.h_der_array[t]) + \
                                  np.matmul(c1_p_y.T, self.c_der_array[t])

        # calc gradient for LSTM parameters (W,U,b for all nodes)
        for i in range(self.timesteps):
            arg_f = np.diagflat((1 - self.f[i]) * self.f[i])
            arg_i = np.diagflat((1 - self.i[i]) * self.i[i])
            arg_o = np.diagflat((1 - self.o[i]) * self.o[i])
            arg_p = np.diagflat((1 - self.p[i] ** 2))

            if i > 0:
                self.W_f_der_array += np.matmul(np.matmul(arg_f, self.f_der_array[i]), self.h[i - 1])
                self.W_i_der_array += np.matmul(np.matmul(arg_i, self.i_der_array[i]), self.h[i - 1])
                self.W_o_der_array += np.matmul(np.matmul(arg_o, self.o_der_array[i]), self.h[i - 1])
                self.W_p_der_array += np.matmul(np.matmul(arg_p, self.p_der_array[i]), self.h[i - 1])

            self.U_f_der_array += np.matmul(np.matmul(arg_f, self.f_der_array[i]), self.x_sequence[i])
            self.U_i_der_array += np.matmul(np.matmul(arg_i, self.i_der_array[i]), self.x_sequence[i])
            self.U_o_der_array += np.matmul(np.matmul(arg_o, self.o_der_array[i]), self.x_sequence[i])
            self.U_p_der_array += np.matmul(np.matmul(arg_p, self.p_der_array[i]), self.x_sequence[i])

            self.b_f_der_array += np.matmul(arg_f.T, self.f_der_array[i])
            self.b_i_der_array += np.matmul(arg_i.T, self.i_der_array[i])
            self.b_o_der_array += np.matmul(arg_o.T, self.o_der_array[i])
            self.b_p_der_array += np.matmul(arg_p.T, self.p_der_array[i])

    def calc_h1_o_y(self, timestep):
        return np.diagflat(np.apply_along_axis(th, 0, self.c[timestep]))

    def calc_h1_p_y(self):
        return np.mamtul(np.matmul(self.diag_o, self.diag_th_c1_der), self.diag_i)

    # calc necessary yakobians:
    def calc_h1_i_y(self):
        return np.matmul(np.matmul(self.diag_o, self.diag_th_c1_der), self.diag_p)

    def calc_h1_f_y(self):
        return np.matmul(np.matmul(self.diag_o, self.diag_th_c1_der), self.diag_c)

    def calc_h_c_y(self, timestep):
        c_t = self.c[timestep - 1]
        diag_th_c_der = np.diagflat(1 - np.apply_along_axis(th, 0, c_t) ** 2)

        return np.matmul(self.diag_o, diag_th_c_der)

    def calc_h1_c_y(self):
        return np.matmul(np.matmul(self.diag_o, self.diag_th_c1_der), self.diag_f)

    def calc_h1_h_y(self, timestep):
        c1_t = self.c[timestep]
        diag_th_c1 = np.diagflat(np.apply_along_axis(th, 0, c1_t))

        return np.matmul(np.matmul(self.diag_o, self.diag_th_c1_der), self.c1_h_y) + \
               np.matmul(np.matmul(diag_th_c1, self.diag_sigma_o_der), self.W_o)

    def calc_c1_h_y(self, timestep):
        c_t = self.c[timestep - 1]

        def calc_f_h_y():
            return np.matmul(self.diag_sigma_f_der, self.W_f)

        def calc_p_h_y():
            return np.matmul(self.diag_th_p_der, self.W_p)

        def calc_i_h_y():
            return np.matmul(self.diag_sigma_i_der, self.W_i)

        return np.matmul(np.diagflat(c_t), calc_f_h_y()) + \
                np.matmul(np.diagflat(self.i[timestep]), calc_i_h_y()) + \
                np.matmul(np.diagflat(self.p[timestep]), calc_p_h_y())



