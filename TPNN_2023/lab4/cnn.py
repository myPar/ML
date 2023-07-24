from TPNN_2023.lab2.activation_functions import ReLU
from TPNN_2023.lab2.tools import uniform_init


def convolute(input_matrix, filter_matrix, stride):
    matrix_size = input_matrix.shape[0]
    filter_size = filter_matrix.shape[0]

    assert matrix_size >= filter_size
    steps_count = (matrix_size - filter_size) // stride + 1
    pass


class CNNlayer:
    def __init__(self, filters_count, filter_size, channels_count, input_shape, stride):
        assert input_shape[0] == channels_count
        input_map_size = input_shape[1]
        assert filter_size <= input_map_size
        self.stride = stride
        self.input_shape = input_shape
        self.filters_shape = (filters_count, channels_count, filter_size, filter_size)

        self.filters = uniform_init(interval=(-0.1, 0.1), array_shape=self.filters_shape)
        self.biases = uniform_init(interval=(-0.1, 0.1), array_shape=(filters_count,))

        self.activation_function = ReLU

        output_map_size = (input_map_size - filter_size) // stride + 1
        self.output_shape = (filters_count, output_map_size, output_map_size)

    def calc_output(self, input_tensor):
        assert input_tensor.shape == self.input_shape
