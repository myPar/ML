import math
import numpy.linalg as la
from TPNN.tools.functions import *
from TPNN.tools.optimizers import Optimizer
from TPNN.tools.data import *


# base layer class
class Layer(object):
    def __init__(self, in_shape, activation_f):
        self.input_shape = in_shape  # shape of input data (for CNN layer it is a 3D array - attributes maps)
        self.output_shape = None  # shape of one unit of output data
        self.x_derivatives_array = None
        self.activation_function = activation_f
        self.input = None
        self.name = None

    def set_x_der_array(self, der_array):
        self.x_derivatives_array = der_array

    def set_out_shape(self, out_sh):
        self.output_shape = out_sh

    def get_output(self, input):
        pass

    def print_config(self):
        pass

    def print_der_config(self):
        pass

    def der_cost_input(self, der_cost_result):  # calculate input derivatives array and returns it
        pass

    def get_x_array(self):
        return self.x_derivatives_array

    def calc_derivatives(self, arg):  # calculates all derivatives: weights, biases, inputs
        pass  # and returns calculated x_der_array
        # arg is either result der array (intermediate layer) or expected output vector (last layer)

    def get_gradient_data(self):
        return None

    def change_parameters(self, apply_function):  # increment net parameters on delta
        pass


class CNNlayer(Layer):
    def __init__(self, in_shape, activation_f, cores_count, cores_shape, stride):
        super().__init__(in_shape, activation_f)
        # check core and input shapes
        assert len(in_shape) == 3 and "invalid input shape, should be a 3D-shape"
        assert len(cores_shape) == 2 and "invalid core's shape"
        assert in_shape[1] >= cores_shape[0] and in_shape[2] >= cores_shape[1] \
               and "core shape is out of input shape's bounds"

        self.name = "CNN layer"

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
        self.biases_derivatives = np.zeros(self.biases.shape)
        self.x_derivatives_array = np.zeros(self.input_shape)

    def accept_filter(self, core_idx, image):
        output_map = np.zeros((self.output_shape[1], self.output_shape[2]))
        filter_core = self.cores[core_idx]

        for i in range(self.output_shape[1]):
            for j in range(self.output_shape[2]):
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
        print(self.name)
        print(" input shape - " + str(self.input_shape) + "; output shape - " + str(self.output_shape))
        print(" cores: ")

        for i in range(len(self.cores)):
            print("[" + str(i) + "]:")
            print_matrix(self.cores[i])

        print(" biases: ")
        print_vector(self.biases)

    def print_der_config(self):
        print(self.name)
        space = "  "
        print(space + "derivatives values:")

        print(space * 2 + "input derivatives:")
        print_arrays(self.x_derivatives_array, space * 3)
        print(space * 2 + "cores derivatives:")
        print_arrays(self.cores_weights_derivatives, space * 3)
        print(space * 2 + "biases derivatives:")
        print_arrays(self.biases_derivatives, space * 3)
        print()

    ##### derivatives calculation:
    ### calc cores' weighs derivatives:
    def get_der_cost_weight(self, core_x, core_y, core_idx, der_cost_result, input_act_maps):
        result_der = 0

        # sum items for each output pixel
        for y in range(self.output_shape[1]):
            for x in range(self.output_shape[2]):
                z = self.z_array[core_idx][y][x]
                term = der_cost_result[core_idx][y][x] * \
                       get_sum_x_from_maps(input_act_maps, y * self.stride + core_y, x * self.stride + core_x) * \
                       get_der(self.activation_function)(z)
                result_der += term

        return result_der

    def der_cost_cores(self, der_cost_result):  # calc dL/dWij for each core
        assert der_cost_result.shape == self.output_shape

        for core_idx in range(len(self.cores)):  # calc der for each core
            core = self.cores[core_idx]

            for core_y in range(core.shape[0]):
                for core_x in range(core.shape[1]):
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
        assert der_cost_result.shape == self.output_shape and "cnn layer"

        for input_map_idx in range(self.input.shape[0]):
            for y in range(self.input.shape[1]):
                for x in range(self.input.shape[2]):
                    result_point_derivative = 0

                    for core_idx in range(len(self.cores)):
                        activation_map = self.input[input_map_idx]
                        result_point_derivative += self.get_der_cost_x(core_idx, activation_map, y, x, der_cost_result)
                    self.x_derivatives_array[input_map_idx][y][x] = result_point_derivative

        return self.x_derivatives_array

    # calc biases derivatives:
    def der_cost_biases(self, der_cost_result):
        assert der_cost_result.shape == self.output_shape

        # biase has linear influence on each pixel of output activation map
        for biase_idx in range(len(self.biases)):
            result_der = 0
            for y in range(self.output_shape[0]):
                for x in range(self.output_shape[1]):
                    result_der += der_cost_result[biase_idx][y][x] * get_der(self.activation_function)(
                        self.z_array[biase_idx][y][x])
            self.biases_derivatives[biase_idx] = result_der

    def calc_derivatives(self, arg):
        self.der_cost_cores(arg)
        self.der_cost_biases(arg)

        return self.der_cost_input(arg)

    def get_gradient_data(self):
        return LayerGradientData(self.cores_weights_derivatives, self.biases_derivatives)

    def change_parameters(self, apply_function):
        self.cores -= apply_function(self.cores_weights_derivatives)
        self.biases -= apply_function(self.biases_derivatives)


class MaxPoolingLayer(Layer):
    def __init__(self, core_shape, in_shape):
        assert len(in_shape) == 3 and "invalid input shape, should be a 3D-shape"
        assert len(core_shape) == 2 and "invalid core shape, should be a 2D-shape"
        assert core_shape[0] <= in_shape[1] and core_shape[1] <= in_shape[2] \
               and "core shape is out of input shape's bounds"

        super().__init__(in_shape, None)
        in_width = in_shape[2]
        in_height = in_shape[1]
        core_width = core_shape[1]
        core_height = core_shape[0]
        self.name = "Max Pooling layer"
        self.core_shape = core_shape
        self.output_shape = (in_shape[0], in_height // core_height, in_width // core_width)

        # list of max cells coordinates (need for back propagation):
        self.input_maximums_positions = np.zeros((self.output_shape[0], self.output_shape[1], self.output_shape[2], 2))
        self.output = np.zeros(self.output_shape)
        self.input = None

    def accept_pooling(self, image, map_idx):
        output = np.zeros((self.output_shape[1], self.output_shape[2]))

        for y in range(self.output_shape[1]):
            for x in range(self.output_shape[2]):
                angle_x = x * self.core_shape[1]
                angle_y = y * self.core_shape[0]

                cur_max_pos = None
                cur_max_value = -100000000

                # get max item from the region bounded by the core:
                for core_y in range(self.core_shape[0]):
                    for core_x in range(self.core_shape[1]):
                        cur_pixel_value = image[angle_y + core_y][angle_x + core_x]

                        if cur_max_value < cur_pixel_value:
                            cur_max_value = cur_pixel_value
                            cur_max_pos = [angle_y + core_y, angle_x + core_x]

                self.input_maximums_positions[map_idx][y][x] = cur_max_pos
                output[y][x] = cur_max_value

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
        print(" maximums positions:")
        print(self.input_maximums_positions)

    def print_der_config(self):
        print(self.name)
        space = "  "
        print(space + "derivatives values:")

        print(space * 2 + "input derivatives:")
        print_arrays(self.x_derivatives_array, space * 3)
        print()

    #### derivatives calculation methods
    def der_cost_input(self, der_cost_result):
        assert der_cost_result.shape == self.output_shape and "max pooling layer"
        assert self.output.shape[0] == self.input_shape[0]
        self.x_derivatives_array = np.zeros(self.input_shape)

        for output_map_idx in range(self.output.shape[0]):
            for y in range(self.output.shape[1]):
                for x in range(self.output.shape[2]):
                    point = self.input_maximums_positions[output_map_idx][y][x]
                    assert point.shape == (2,)
                    # tipping gradients throw polling layer:
                    self.x_derivatives_array[output_map_idx][int(point[0])][int(point[1])] = der_cost_result[output_map_idx][y][x]

        return self.x_derivatives_array

    def calc_derivatives(self, arg):
        return self.der_cost_input(arg)


class DenseLayer(Layer):
    def __init__(self, activation_f, neuron_count, prev_layer_neuron_count):
        super().__init__((neuron_count,), activation_f)
        self.output_shape = (neuron_count,)
        self.input_shape = (prev_layer_neuron_count,)
        self.neuron_count = neuron_count
        self.name = "Dense layer"
        # init weighs matrix (Wij) - i - idx of neuron from prev layer j - idx of neuron form cur layer
        self.weights_matrix = np.random.rand(prev_layer_neuron_count, neuron_count) * 2 - 1
        self.biases = np.random.rand(neuron_count, 1)  # vector-column of biases
        self.z_array = None

        # derivatives fields:
        self.weights_der_array = np.zeros(self.weights_matrix.shape)
        self.biases_der_array = np.zeros(self.biases.shape)
        self.x_derivatives_array = np.zeros(self.input_shape)

    def set_weighs(self, weight_matrix):
        assert weight_matrix.shape == (self.input_shape[0], self.output_shape[0]) and "invalid weight matrix shape"
        self.weights_matrix = weight_matrix

    def set_biases(self, biases):
        assert biases.shape == self.biases.shape and "invalid biases vector shape"
        self.biases = biases

    def get_output(self, input):
        assert len(input.shape) == 1 and "invalid input shape, should be 1-d array"
        assert input.shape == self.input_shape and "input data shape doesn't match with laler's shape"

        self.input = input  # cache input
        self.z_array = np.matmul(self.weights_matrix.transpose(), np.array([input]).transpose()) + self.biases
        assert self.z_array.shape == (self.output_shape[0], 1) and "invalid output vector shape"

        return np.apply_along_axis(self.activation_function, 0, self.z_array).reshape(self.output_shape)

    def print_config(self):
        print("Dense layer")
        print(" neuron count: " + str(self.neuron_count))
        print(" input shape - " + str(self.input_shape) + "; output shape - " + str(self.output_shape))
        print(" weight matrix:")
        print_matrix(self.weights_matrix)
        print(" biases vector: ")
        print_vector(self.biases)

    def print_der_config(self):
        print(self.name)
        space = "  "
        print(space + "derivatives values:")

        print(space * 2 + "input derivatives:")
        print_arrays(self.x_derivatives_array, space * 3)
        print(space * 2 + "weights derivatives:")
        print_arrays(self.weights_der_array, space * 3)
        print(space * 2 + "biases derivatives:")
        print_arrays(self.biases_der_array, space * 3)
        print()

    ### calc derivatives methods:
    def der_cost_weights(self, der_cost_result):
        assert der_cost_result.shape == self.output_shape and "invalid Y der array shape"

        for i in range(self.weights_matrix.shape[0]):
            for j in range(self.weights_matrix.shape[1]):
                self.weights_der_array[i][j] = der_cost_result[j] * get_der(self.activation_function)(
                    self.z_array[j][0]) * self.input[i]

    def der_cost_input(self, der_cost_result):
        assert der_cost_result.shape == self.output_shape and "dense layer: invalid Y der array shape"
        prev_layer_neuron_count = self.input_shape[0]
        cur_layer_neuron_count = self.neuron_count

        for prev_layer_neuron_idx in range(prev_layer_neuron_count):
            result_der = 0
            # each neuron from previous layer linearly affects on each neuron on current layer
            for cur_layer_neuron_idx in range(cur_layer_neuron_count):
                result_der += der_cost_result[cur_layer_neuron_idx] * \
                              get_der(self.activation_function)(self.z_array[cur_layer_neuron_idx]) \
                              * self.weights_matrix[prev_layer_neuron_idx][cur_layer_neuron_idx]
            self.x_derivatives_array[prev_layer_neuron_idx] = result_der

        return self.x_derivatives_array

    def der_cost_biases(self, der_cost_result):
        assert der_cost_result.shape == self.output_shape and "invalid Y der array shape"

        for b_idx in range(len(self.biases)):
            self.biases_der_array[b_idx][0] = der_cost_result[b_idx] * \
                                              get_der(self.activation_function)(self.z_array[b_idx][0])

    def calc_derivatives(self, arg):
        self.der_cost_weights(arg)
        self.der_cost_biases(arg)

        return self.der_cost_input(arg)

    def get_gradient_data(self):
        return LayerGradientData(self.weights_der_array, self.biases_der_array)

    def change_parameters(self, apply_function):
        self.weights_matrix -= apply_function(self.weights_der_array)
        self.biases -= apply_function(self.biases_der_array)


# flatten attribute's maps and concatenate it in one big vector. To apply it in Dense layer input
class ReformatLayer(Layer):
    def __init__(self, input_shape):
        assert len(input_shape) == 3 and "invalid input shape, should be a 3d"
        super().__init__(input_shape, None)

        map_shape = (input_shape[1], input_shape[2])
        self.output_shape = (get_size(input_shape),)
        self.name = "Reformat layer"

    def get_output(self, input):
        assert input.shape == self.input_shape and "invalid input data shape"
        result = input[0].flatten()

        # flatten attributes maps:
        for i in range(input.shape[0] - 1):
            map_idx = i + 1
            cur_map = input[map_idx]
            result = np.concatenate((result, cur_map), axis=None)
        return result

    def print_config(self):
        print("Reformat layer")
        print(" input shape - " + str(self.input_shape) + "; output shape - " + str(self.output_shape))

    ### calc derivatives methods:
    def der_cost_input(self, der_cost_result):
        assert der_cost_result.shape == self.output_shape and "reformat layer"
        self.x_derivatives_array = der_cost_result.reshape(self.input_shape)

        return self.x_derivatives_array

    def calc_derivatives(self, arg):
        return self.der_cost_input(arg)


class SoftmaxLayer(Layer):
    def __init__(self, neuron_count):
        super().__init__((neuron_count,), None)
        self.neuron_count = neuron_count
        self.output_shape = (neuron_count,)
        self.x_derivatives_array = np.zeros(self.output_shape)
        self.output = None
        self.loss_function = log_loss
        self.name = "Softmax layer"

    def get_output(self, input):
        assert len(input.shape) == 1 and "should be 1d vector"
        assert len(input) == self.neuron_count and "invalid input items count"

        denominator = np.sum(np.exp(input))
        self.output = np.array([np.exp(input[i]) / denominator for i in range(len(input))])

        return self.output

    def print_config(self):
        print("Softmax layer")
        print(" input shape - " + str(self.input_shape) + "; output shape - " + str(self.output_shape))

    def print_der_config(self):
        print(self.name)
        space = "  "
        print(space + "derivatives values:")

        print(space * 2 + "input derivatives:")
        print_arrays(self.x_derivatives_array, space * 3)
        print()

    def der_cost_input(self, actual_vector):
        assert actual_vector.shape == self.output_shape and "softmax layer: invalid shape for Y der array"
        self.x_derivatives_array = self.output - actual_vector

        # already calculated that dY/dx = Y - t:
        return self.x_derivatives_array

    def calc_derivatives(self, arg):
        return self.der_cost_input(arg)


# Net as stack of layers
class Net(object):
    def __init__(self):
        self.layers = []
        self.layer_count = 0
        self.learning_rate = 0.001
        self.optimizer = None
        self.need_debug = False
        self.loss_list = []
        self.metric_list = []
        self.print_train_metrics = True

    def get_last_layer(self):
        return self.layers[self.layer_count - 1]

    # change net configuration methods:
    def add_layer(self, layer: Layer):
        if self.layer_count > 0:
            # layer's compatibility check
            last_layer = self.get_last_layer()
            assert last_layer.output_shape == layer.input_shape and "layer shapes are incompatible"
        self.layers.append(layer)
        self.layer_count += 1

    def set_learning_rate(self, lr):
        assert lr > 0
        self.learning_rate = lr

    def set_optimizer(self, optimizer: Optimizer):
        self.optimizer = optimizer

    def debug(self):
        if self.need_debug:
            for layer in self.layers:
                layer.print_der_config()

    # calculate summarized gradient norm for all net
    def get_net_gradients_norm(self):
        result_norm = 0

        for layer in self.layers:
            grad_data = layer.get_gradient_data()

            if not (grad_data is None):
                w_grad_norm = la.norm(grad_data.weights_gradient)
                b_grad_norm = la.norm(grad_data.biases_gradient)

                result_norm += (w_grad_norm ** 2 + b_grad_norm ** 2)

        return math.sqrt(result_norm)

    # net learning methods:
    def change_parameters(self):
        gradient_norm = self.get_net_gradients_norm()
        coefficient = self.optimizer.get_next_coefficient(gradient_norm)

        # iterate over net layers and change weights and biases:
        for layer in self.layers:
            l = lambda x: self.learning_rate * coefficient * x
            f = np.vectorize(l)
            layer.change_parameters(f)

    def step(self, input_item, expected_output_item):
        # forward propagation:
        output = self.get_output(input_item)

        # get input derivatives array on last layer:
        last_layer = self.get_last_layer()
        der_cost_result = last_layer.calc_derivatives(expected_output_item)

        # calc derivatives in other layers:
        for layer_idx in range(self.layer_count - 2, -1, -1):
            cur_layer = self.layers[layer_idx]
            der_cost_result = cur_layer.calc_derivatives(der_cost_result)
        self.debug()
        # change net parameters:
        self.change_parameters()

        return output

    def get_predictions(self, input_data):
        assert input_data.shape[1:] == self.layers[0].input_shape
        predictions = []

        for item in input_data:
            predictions.append(self.get_output(item))

        return np.array(predictions)

    def train(self, data: Data, epoch_count):  # net training on given data
        self.loss_list = []
        self.metric_list = []

        for epoch in range(epoch_count):
            predictions = []

            for item_idx in range(data.size):
                output = self.step(data.input_data[item_idx], data.expected_output_data[item_idx])
                predictions.append(output)
                #print("step-" + str(item_idx))
            predictions = np.array(predictions)

            loss = average_loss(data.expected_output_data, predictions, log_loss)
            metric = categorical_accuracy(data.expected_output_data, predictions)

            self.loss_list.append(loss)
            self.metric_list.append(metric)

            if self.print_train_metrics:
                print("epoch-" + str(epoch) + ": loss=" + str(loss) + "; metric=" + str(metric))

    # other:
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


def der_softmax_x(y_idx, x_idx, y_vector):
    if y_idx == x_idx:
        return y_vector[y_idx] - y_vector[y_idx] ** 2
    else:
        return -y_vector[y_idx] * y_vector[x_idx]
