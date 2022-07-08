import numpy as np
from numpy import random


class LSTM(object):
    def __init__(self, input_shape, output_size, timesteps):
        self.name = 'lstm'
        input_dim = len(input_shape)

        assert 1 <= input_dim <= 2 and self.name + " - invalid input shape dim"

        # init base fields:
        self.timesteps = timesteps
        self.x_size = input_shape[0] if input_dim == 1 else input_shape[0] * input_shape[1]
        self.h_size = output_size

        # init net parameters:
        self.U_i = random.rand(self.h_size, self.x_size)
        self.U_f = random.rand(self.h_size, self.x_size)
        self.U_p = random.rand(self.h_size, self.x_size)
        self.U_o = random.rand(self.h_size, self.x_size)

        self.W_i = random.rand(self.h_size, self.h_size)
        self.W_f = random.rand(self.h_size, self.h_size)
        self.W_p = random.rand(self.h_size, self.h_size)
        self.W_o = random.rand(self.h_size, self.h_size)

        self.b_i = random.rand(self.h_size, 1)
        self.b_f = random.rand(self.h_size, 1)
        self.b_p = random.rand(self.h_size, 1)
        self.b_o = random.rand(self.h_size, 1)

        # init lambda set (i,f,p,o) for each timestep:
        self.i = np.zeros((timesteps, self.h_size, 1))
        self.f = np.zeros((timesteps, self.h_size, 1))
        self.p = np.zeros((timesteps, self.h_size, 1))
        self.o = np.zeros((timesteps, self.h_size, 1))

        # init h and c value's arrays:
        self.h = np.zeros((timesteps, self.h_size, 1))
        self.c = np.zeros((timesteps, self.h_size, 1))

        # init gradients (h, c, lambda set, net parameters derivatives):
        self.i_der_array = np.zeros((timesteps, self.h_size, 1))
        self.f_der_array = np.zeros((timesteps, self.h_size, 1))
        self.p_der_array = np.zeros((timesteps, self.h_size, 1))
        self.o_der_array = np.zeros((timesteps, self.h_size, 1))

        self.h_der_array = np.zeros((timesteps, self.h_size, 1))
        self.c_der_array = np.zeros((timesteps, self.h_size, 1))

        self.W_i_der_array = np.zeros((self.h_size, self.h_size))
        self.W_f_der_array = np.zeros((self.h_size, self.h_size))
        self.W_p_der_array = np.zeros((self.h_size, self.h_size))
        self.W_o_der_array = np.zeros((self.h_size, self.h_size))

        self.U_i_der_array = np.zeros((self.h_size, self.x_size))
        self.U_f_der_array = np.zeros((self.h_size, self.x_size))
        self.U_p_der_array = np.zeros((self.h_size, self.x_size))
        self.U_o_der_array = np.zeros((self.h_size, self.x_size))

        self.b_i_der_array = np.zeros((self.h_size, 1))
        self.b_f_der_array = np.zeros((self.h_size, 1))
        self.b_p_der_array = np.zeros((self.h_size, 1))
        self.b_o_der_array = np.zeros((self.h_size, 1))
