import numpy as np
from TPNN.tools.activation_functions import *


# base layer class
class Layer(object):
    def __init__(self, in_shape, activation_f):
        self.input_shape = in_shape  # shape of input data (for CNN layer it is a 3D array - attributes maps)
        self.output_shape = None  # shape of one unit of output data
        self.x_derivatives_array = None
        self.activation_function = activation_f
        self.input = None

    def set_x_der_array(self, der_array):
        self.x_derivatives_array = der_array

    def set_out_shape(self, out_sh):
        self.output_shape = out_sh

    def get_output(self, input):
        pass

    def print_config(self):
        pass


class CNNlayer(Layer):
    def __init__(self, in_shape, activation_f, cores_count, cores_shape, stride):
        super().__init__(in_shape, activation_f)
        # check core and input shapes
        assert len(in_shape) == 3 and "invalid input shape, should be a 3D-shape"
        assert len(cores_shape) == 2 and "invalid core's shape"
        assert in_shape[1] >= cores_shape[0] and in_shape[2] >= cores_shape[1] \
               and "core shape is out of input shape's bounds"

        self.stride = stride
        in_width = in_shape[2]
        in_height = in_shape[1]

        core_width = cores_shape[1]
        core_height = cores_shape[0]

        self.output_shape = (
            cores_count, (in_height - core_height) // stride + 1, (in_width - core_width) // stride + 1)
        self.output = np.zeros(self.output_shape)
        self.z_array = np.zeros(self.output_shape)

        # init weights in cores and biases in [-1,1] interval
        self.cores = np.random.rand(cores_count, cores_shape[0], cores_shape[1]) * 2 - 1
        self.biases = np.random.rand(cores_count) * 2 - 1

        # derivatives fields:
        self.cores_weights_derivatives = np.zeros(self.cores.shape)
        self.input_pixels_derivatives = np.zeros(self.input_shape)

    def accept_filter(self, core_idx, image):
        output_map = np.zeros((self.output_shape[1], self.output_shape[2]))
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
                output_map[i][j] = z

        return output_map

    def get_output(self, input):
        assert input.shape == self.input_shape and "input shape doesn't match with layer input shape"
        assert len(input.shape) == 3 and "input shape should have 3 dimensions"
        self.input = input  # cache input data
        self.z_array = np.zeros(self.output_shape)  # reset z-array
        map_count = input.shape[0]  # count of attribute's map

        # accept each filter to all attribute maps and concatenate the result
        for core_idx in range(len(self.cores)):
            result_map = np.zeros((self.output_shape[1], self.output_shape[2]))

            for map_idx in range(map_count):
                result_map += self.accept_filter(core_idx, input[map_idx])
            self.z_array[core_idx] = result_map + self.biases[core_idx]
            self.output[core_idx] = np.apply_along_axis(self.activation_function, 0, self.z_array[core_idx])

        return self.output

    def set_cores(self, cores):
        assert len(cores) == len(self.cores) and "invalid core's count"
        assert cores[0].shape == self.cores[0].shape and "invalid core's shape"
        self.cores = cores

    def set_biases(self, biases):
        assert self.biases.shape == biases.shape and "invalid biases shape"
        self.biases = biases

    def print_config(self):
        print("CNN layer:")
        print(" input shape - " + str(self.input_shape) + "; output shape - " + str(self.output_shape))
        print(" cores: ")

        for i in range(len(self.cores)):
            print("[" + str(i) + "]:")
            print_matrix(self.cores[i])

        print(" biases: ")
        print_vector(self.biases)

    ##### derivatives calculation:
    ### calc cores' weighs derivatives:
    def get_der_cost_weight(self, core_x, core_y, core_idx, der_cost_result, input_act_maps):
        result_der = 0

        # sum items for each output pixel
        for y in range(self.output_shape[1]):
            for x in range(self.output_shape[2]):
                z = self.z_array[core_idx][y][x]
                term = der_cost_result[core_idx][y][x] * get_sum_x_from_maps(input_act_maps, y + core_y, x + core_x) \
                       * get_der(self.activation_function)(z)
                result_der += term

        return result_der

    def der_cost_cores(self, der_cost_result):  # calc dL/dWij for each core
        for core_idx in range(len(self.cores)):  # calc der for each core
            core = self.cores[core_idx]

            for core_y in range(len(core.shape[0])):
                for core_x in range(len(core.shape[1])):
                    self.cores_weights_derivatives[core_idx][core_y][core_x] = \
                        self.get_der_cost_weight(core_x, core_y, core_idx, der_cost_result, self.input)

    ### calc input pixels derivatives:
    def get_der_cost_x(self, core_idx, input_act_map, y, x, der_cost_result):
        left_up_point, right_down_point = get_bound_point(input_act_map.shape, self.cores[0].shape, x, y)
        result_der = 0
        core = self.cores[core_idx]

        # 'core_y_on_map' and 'core_x_on_map' - coordinates of the left up core point on the image
        for core_y_on_map in np.arange(left_up_point[0], right_down_point[0]):
            for core_x_on_map in np.arange(left_up_point[1], right_down_point[1]):
                core_y = y - core_y_on_map  # calc coordinates of weight on core which x point is collided
                core_x = x - core_x_on_map  #
                result_der += der_cost_result[core_idx][core_y_on_map][core_x_on_map] * core[core_y][core_x] * \
                              get_der(self.activation_function)(self.z_array[core_idx][core_y_on_map][core_x_on_map])

        return result_der

    def der_cost_input(self, der_cost_result):
        for input_map_idx in range(self.input.shape[0]):
            for y in range(self.input.shape[1]):
                for x in range(self.input.shape[2]):
                    result_point_derivative = 0

                    for core_idx in range(len(self.cores)):
                        activation_map = self.input[input_map_idx]
                        result_point_derivative += self.get_der_cost_x(core_idx, activation_map, y, x, der_cost_result)
                    self.input_pixels_derivatives[input_map_idx][y][x] = result_point_derivative


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
        self.output_shape = (in_shape[0], in_height // core_height, in_width // core_width)

        # list of max cells coordinates (need for back propagation):
        self.input_maximums_positions = np.zeros((self.output_shape[0], self.output_shape[1] * self.output_shape[2], 2))
        self.output = np.zeros(self.output_shape)

    def accept_pooling(self, image, map_idx):
        iteration = 0
        output = np.zeros((self.output_shape[1], self.output_shape[2]))

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

                self.input_maximums_positions[map_idx][iteration] = cur_max_pos
                iteration += 1
                output[i][j] = cur_max_value

        return output

    def get_output(self, input):
        assert input.shape == self.input_shape and "input shape doesn't match with layer input shape"
        assert len(input.shape) == 3 and "input shape should have 3 dimensions"

        map_count = input.shape[0]

        for map_idx in range(map_count):
            self.output[map_idx] = self.accept_pooling(input[map_idx], map_idx)

        return self.output

    def print_config(self):
        print("MaxPooling layer")
        print(" input shape - " + str(self.input_shape) + "; output shape - " + str(self.output_shape))
        print(" core shape: " + str(self.core_shape))


class DenseLayer(Layer):
    def __init__(self, activation_f, neuron_count, prev_layer_neuron_count):
        super().__init__((neuron_count,), activation_f)
        self.output_shape = (neuron_count,)
        self.input_shape = (prev_layer_neuron_count,)
        self.neuron_count = neuron_count
        # init weighs matrix (Wij) - i - idx of neuron from prev layer j - idx of neuron form cur layer
        self.weights_matrix = np.random.rand(prev_layer_neuron_count, neuron_count) * 2 - 1
        self.biases = np.random.rand(neuron_count, 1)  # vector-column of biases

    def set_weighs(self, weight_matrix):
        assert weight_matrix.shape == (self.input_shape[0], self.output_shape[0]) and "invalid weight matrix shape"
        self.weights_matrix = weight_matrix

    def set_biases(self, biases):
        assert biases.shape == self.biases.shape and "invalid biases vector shape"
        self.biases = biases

    def get_output(self, input):
        assert len(input.shape) == 1 and "invalid input shape, should be 1-d array"
        assert input.shape == self.input_shape and "input data shape doesn't match with laler's shape"

        z_array = np.matmul(self.weights_matrix.transpose(), np.array([input]).transpose()) + self.biases
        assert z_array.shape == (self.output_shape[0], 1) and "invalid output vector shape"

        return np.apply_along_axis(self.activation_function, 0, z_array).reshape(self.output_shape)

    def print_config(self):
        print("Dense layer")
        print(" neuron count: " + str(self.neuron_count))
        print(" input shape - " + str(self.input_shape) + "; output shape - " + str(self.output_shape))
        print(" weight matrix:")
        print_matrix(self.weights_matrix)
        print(" biases vector: ")
        print_vector(self.biases)


# concatenate attribute's maps and reshape it. To apply it in Dense layer input
class ReformatLayer(Layer):
    def __init__(self, input_shape):
        assert len(input_shape) == 3 and "invalid input shape, should be a 3d"
        super().__init__(input_shape, None)

        map_shape = (input_shape[1], input_shape[2])
        self.output_shape = (get_size(map_shape),)

    def get_output(self, input):
        assert input.shape == self.input_shape and "invalid input data shape"
        # concatenate attribute's maps
        result_map = np.zeros((input.shape[1], input.shape[2]))

        for cur_map in input:
            result_map += cur_map

        return result_map.reshape(self.output_shape)

    def print_config(self):
        print("Reformat layer")
        print(" input shape - " + str(self.input_shape) + "; output shape - " + str(self.output_shape))


class SoftmaxLayer(Layer):
    def __init__(self, neuron_count):
        super().__init__((neuron_count,), None)
        self.neuron_count = neuron_count
        self.output_shape = (neuron_count,)

    def get_output(self, input):
        assert len(input.shape) == 1 and "should be 1d vector"
        assert len(input) == self.neuron_count and "invalid input items count"

        denominator = np.sum(np.exp(input))

        return np.array([np.exp(input[i]) / denominator for i in range(len(input))])

    def print_config(self):
        print("Softmax layer")
        print(" input shape - " + str(self.input_shape) + "; output shape - " + str(self.output_shape))


# Net as stack of layers
class Net(object):
    def __init__(self):
        self.layers = []
        self.layer_count = 0

    def get_last_layer(self):
        return self.layers[self.layer_count - 1]

    def add_layer(self, layer: Layer):
        if self.layer_count > 0:
            # layer's compatibility check
            last_layer = self.get_last_layer()
            assert last_layer.output_shape == layer.input_shape and "layer shapes are incompatible"
        self.layers.append(layer)
        self.layer_count += 1

    def get_output(self, input_data):
        input = input_data

        for layer in self.layers:
            input = layer.get_output(input)
        return input

    def print_config(self):
        print("Layers count: " + str(self.layer_count))
        print("layers configuration: ")
        print()

        for layer in self.layers:
            layer.print_config()
            print()


def get_size(sh: tuple):
    result = 1
    for i in range(len(sh)):
        result *= sh[i]

    return result


def print_matrix(matrix):
    assert len(matrix.shape) == 2

    for i in range(matrix.shape[0]):
        print("     ", end='')

        for j in range(matrix.shape[1]):
            print(str(matrix[i][j]) + " ", end='')
        print()


def print_vector(vector):
    dim_count = len(vector.shape)
    assert dim_count == 1 or dim_count == 2 and "should be a vector-column or a vector-row or 1-d vector"

    print("     ", end='')
    display_vector = vector.reshape((get_size(vector.shape),))

    for i in display_vector:
        print(str(i) + " ", end='')
    print()


def get_sum_x_from_maps(maps, y, x):
    result = 0

    for map in maps:
        result += map[y][x]

    return result


def get_bound_point(image_shape, core_shape, point_x, point_y):
    im_width = image_shape[1]
    im_height = image_shape[0]

    core_width = core_shape[1]
    core_height = core_shape[0]

    left_up_y = max(point_y - core_height + 1, 0)
    left_up_x = max(point_x - core_width + 1, 0)

    right_down_x = min(point_x, im_width - core_width)
    right_down_y = min(point_y, im_height - core_height)

    return (left_up_y, left_up_x), (right_down_y, right_down_x)  # (y,x) notation
