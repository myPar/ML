import numpy as np
import numpy.linalg as la


# activation functions
def ident(x):
    return x


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def th(x):
    return np.tanh(x)


def ELU(x, alpha):
    if x >= 0:
        return x
    else:
        return alpha * (np.exp(x) - 1)


# derivatives of activation functions
def ident_der(x):
    return 1


def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x))


def th_der(x):
    return 1 - th(x) ** 2


def ELU_der(x, alpha):
    if x >= 0:
        return 1
    else:
        return alpha * np.exp(x)


def get_der(function_ident):
    if function_ident == sigmoid:
        return sigmoid_der
    elif function_ident == ident:
        return ident_der
    elif function_ident == th:
        return th_der
    elif function_ident == ELU:
        return ELU_der
    else:
        assert False


class Layer(object):
    def __init__(self, neuron_count, act_function):
        assert neuron_count > 0

        self.neuron_count = neuron_count  # count of neurons in layer
        self.activations = np.array([np.zeros(neuron_count)])  # fill activations with zeros
        self.z_array = np.array([np.zeros(neuron_count)])
        self.act_function = act_function
        self.weights = None  # weight matrix (Wij - weigh from i'th neuron of k-1 layer to
        ## j-th neuron of k layer)
        self.biases = np.array([np.zeros(neuron_count)])  # fill bias array with zeros

    def set_weights(self, weight_matrix):
        shape = weight_matrix.shape[1]
        assert shape == self.neuron_count and "invalid weigh matrix shape - " + str(shape)
        self.weights = weight_matrix

    def set_bias(self, biases):
        shape = len(biases[0])
        assert shape == self.neuron_count and "invalid bias array shape - " + str(shape)
        self.biases = biases

    def print_layer_config(self):
        print(" size=" + str(self.neuron_count))
        print(" act_function=" + str(self.act_function))
        print(" activations=" + str(self.activations))
        print(" biases=" + str(self.biases))
        print(" weights:")
        if self.weights is None:
            print("     empty", end='')
        else:
            for line in self.weights:
                print("     |", end='')
                for item in line:
                    print(str(item) + " ", end='')
                print("|")
        print()
        print("--------------------------")


class Net(object):
    def __init__(self):
        self.layers = []
        self.layers_count = 0

    def clear_weights(self, layer_idx: int):
        layers_count = len(self.layers)
        assert 0 <= layer_idx <= layers_count - 1 and "layer idx - " + str(layer_idx)
        self.layers[layer_idx].set_weights(None)

    def insert_layer(self, idx: int, layer_size: int, act_function):
        assert 0 <= idx <= self.layers_count and "invalid insert layer idx - " + str(idx)

        if idx != self.layers_count:  # clear weights of behind layer
            self.clear_weights(idx)
        self.layers.insert(idx, Layer(layer_size, act_function))
        self.layers_count += 1

    def init_weights(self, layer_idx, weights):
        assert 0 < layer_idx < self.layers_count and "index is out of layer's bounds - " + str(layer_idx)
        assert weights.shape[1] == self.layers[layer_idx].neuron_count and \
               weights.shape[0] == self.layers[layer_idx - 1].neuron_count and \
               "invalid shape weights matrix shape - " + str(weights.shape)
        self.layers[layer_idx].set_weights(weights)

    def init_biases(self, layer_idx: int, biases):
        bias_length = len(biases[0])
        assert bias_length == self.layers[layer_idx].neuron_count and "invalid bias vector size - " + str()
        self.layers[layer_idx].set_bias(biases)

    def calc_output(self, input_data):
        first_layer = self.layers[0]
        assert len(input_data) == first_layer.neuron_count

        # set activations of the first layer to input data
        for i in range(first_layer.neuron_count):
            first_layer.activations[0][i] = input_data[i]

        # calc activations on each layer
        for i in range(self.layers_count - 1):
            idx = i + 1
            cur_layer = self.layers[idx]
            prev_layer = self.layers[idx - 1]

            w_matrix = cur_layer.weights
            prev_layer_activations = prev_layer.activations

            z = np.matmul(w_matrix.transpose(), prev_layer_activations.transpose()) + cur_layer.biases.transpose()
            z = z.transpose()
            cur_layer.z_array = z

            cur_layer.activations = np.apply_along_axis(cur_layer.act_function, 0, z)
        # output is last layer's activations:
        return self.layers[self.layers_count - 1].activations

    # ------------------------#
    # stochastic gradient descent methods:

    # derivative of cost function on activation
    def der_cost_act(self, layer_idx, neuron_idx, next_layer_act_der):
        assert 0 <= layer_idx < self.layers_count - 1 and "out of bounds layer idx"
        next_layer = self.layers[layer_idx + 1]
        result = 0

        for i in range(next_layer.neuron_count):
            item = next_layer_act_der[i]
            result += (item * get_der(next_layer.act_function)(next_layer.z_array[0][i]) *
                       next_layer.weights[neuron_idx][i])

        return result

    # derivative of const function on weigh
    def der_cost_weigh(self, layer_idx, i, j, cost_act_der_value):
        assert 0 < layer_idx < self.layers_count

        cur_layer = self.layers[layer_idx]
        prev_layer = self.layers[layer_idx - 1]

        return cost_act_der_value * get_der(cur_layer.act_function)(cur_layer.z_array[0][j]) * \
               prev_layer.activations[0][i]

    # derivative of cost function on bias
    def der_cost_bias(self, layer_idx, neuron_idx, cost_act_der_value):
        assert 0 < layer_idx < self.layers_count
        cur_layer = self.layers[layer_idx]

        return cost_act_der_value * get_der(cur_layer.act_function)(cur_layer.z_array[0][neuron_idx])

    # --------------------#
    # other methods:
    def print_net_config(self):
        print("layers=" + str(self.layers_count))

        i = 0
        for layer in self.layers:
            print("[" + str(i) + "]")
            layer.print_layer_config()
            i += 1


def predict_error(out, target):
    return la.norm(out - target, 'fro')  # l2 norm


def predict_error_der(out, target, neuron_idx):
    return out[neuron_idx] - target[neuron_idx]
