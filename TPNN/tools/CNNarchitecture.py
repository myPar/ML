import numpy as np


# base layer class
class Layer(object):
    def __init__(self, in_shape, activation_f):
        self.input_shape = in_shape  # shape of input data (for CNN layer it is a 3D array - attributes maps)
        self.output_shape = None  # shape of one unit of output data
        self.x_derivatives_array = None
        self.activation_function = activation_f

    def set_x_der_array(self, der_array):
        self.x_derivatives_array = der_array

    def set_out_shape(self, out_sh):
        self.output_shape = out_sh


class CNNlayer(Layer):
    def __init__(self, in_shape, activation_f, cores_count, cores_shape, stride):
        super().__init__(in_shape, activation_f)
        # check core and input shapes
        assert len(in_shape) == 3 and "invalid input shape, should be a 3D-shape"
        assert len(cores_shape) == 2 and "invalid core's shape"
        assert in_shape[1] >= cores_shape[0] and in_shape[2] >= cores_shape[1]\
               and "core shape is out of input shape's bounds"

        self.stride = stride
        in_width = in_shape[2]
        in_height = in_shape[1]

        core_width = cores_shape[1]
        core_height = cores_shape[0]

        self.output_shape = (cores_count, (in_height - core_height) / stride + 1, (in_width - core_width) / stride + 1)
        self.output = np.zeros(self.output_shape)

        # init weights in cores and biases in [-1,1] interval
        self.cores = np.random.rand(cores_count, cores_shape[0], cores_shape[1]) * 2 - 1
        self.biases = np.random.rand(cores_count) * 2 - 1

    def accept_filter(self, core_idx, image):
        output_map = np.zeros(self.output_shape[0], self.output_shape[1])
        filter_core = self.cores[core_idx]

        for i in range(self.output_shape[0]):
            for j in range(self.output_shape[1]):
                angle_x = j * self.stride
                angle_y = i * self.stride

                # mul elements:
                z = 0
                for core_y in range(filter_core.shape[0]):
                    for core_x in range(filter_core.shape[1]):
                        z += filter_core[core_y][core_x] * image[angle_y + core_y][angle_x + core_x]
                output_map[i][j] = self.activation_function(z + self.biases[core_idx])

        return output_map

    def get_output(self, input):
        assert input.shape == self.input_shape and "input shape doesn't match with layer input shape"
        assert len(input.shape) == 3 and "input shape should have 3 dimensions"

        map_count = input.shape[0]  # count of attribute's map

        # accept each filter to all attribute maps and concatenate the result
        for core_idx in range(len(self.cores)):
            result_map = np.zeros(self.output_shape[1], self.output_shape[2])

            for map_idx in range(map_count):
                result_map += self.accept_filter(core_idx, input[map_idx])
            self.output[core_idx] = result_map

        return self.output


class MaxPoolingLayer(Layer):
    def __init__(self, core_shape, in_shape):
        assert len(in_shape) == 3 and "invalid input shape, should be a 3D-shape"
        assert len(core_shape) == 2 and "invalid core shape, should be a 2D-shape"
        assert core_shape[0] <= in_shape[1] and core_shape[1] <= in_shape[
            2] and "core shape is out of input shape's bounds"

        super().__init__(in_shape, None)
        in_width = in_shape[2]
        in_height = in_shape[1]
        core_width = core_shape[1]
        core_height = core_shape[0]
        self.core_shape = core_shape
        self.output_shape = (in_shape[0], in_width / core_width, in_height / core_height)
        self.input_maximums_positions = np.zeros(core_width * core_height, 2)  # list of max cells coordinates (need for back propagation)
        self.output = np.zeros(self.output_shape)

    def accept_pooling(self, image):
        iteration = 0
        output = np.zeros(self.output_shape)

        for i in range(self.output_shape[1]):
            for j in range(self.output_shape[2]):
                angle_x = j * self.core_shape[1]
                angle_y = i * self.core_shape[0]

                cur_max_pos = None
                cur_max_value = -100000000

                # get max item from the region bounded by the core:
                for core_y in range(self.core_shape[0]):
                    for core_x in range(self.core_shape[1]):
                        cur_pixel_value = image[angle_y + core_y][angle_x + core_x]

                        if cur_max_value < cur_pixel_value:
                            cur_max_value = cur_pixel_value
                            cur_max_pos = [angle_y + core_y, angle_x + core_x]

                self.input_maximums_positions[iteration] = cur_max_pos
                iteration += 1
                output[i][j] = cur_max_value

        return output

    def get_output(self, input):
        assert input.shape == self.input_shape and "input shape doesn't match with layer input shape"
        assert len(input.shape) == 3 and "input shape should have 3 dimensions"

        map_count = input.shape[0]

        for map_idx in range(map_count):
            self.output[map_idx] = self.accept_pooling(input[map_idx])

        return self.output


class DenseLayer(Layer):
    def __init__(self, activation_f, neuron_count, prev_layer_neuron_count):
        super().__init__((neuron_count,), activation_f)
        self.output_shape = (neuron_count,)
        self.input_shape = (prev_layer_neuron_count,)
        # init weighs matrix (Wij) - i - idx of neuron from prev layer j - idx of neuron form cur layer
        self.weights_matrix = np.random.rand((prev_layer_neuron_count, neuron_count)) * 2 - 1
        self.biases_vector = np.random.rand((neuron_count, 1)) # vector-column of biases

    def get_output(self, input):
        assert len(input.shape) == 1 and "invalid input shape, should be 1-d array"
        assert input.shape == self.input_shape and "input data shape doesn't match with laler's shape"

        z_array = np.matmul(self.weights_matrix.transpose(), np.array([input])) + self.biases_vector
        assert z_array.shape == (self.output_shape[0], 1) and "invalid output vecor shape"

        return np.apply_along_axis(self.activation_function, 0, z_array).reshape(self.output_shape)