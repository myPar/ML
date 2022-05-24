class Data(object):
    def __init__(self, input_data, expected_output_data):
        assert len(input_data) == len(expected_output_data)
        assert len(input_data.shape) > 1 and len(expected_output_data.shape) > 1

        self.input_data = input_data                        # net input data
        self.expected_output_data = expected_output_data    # what expected on net output
        self.size = len(expected_output_data)               # number of items in this unit of data

        self.input_shape = input_data[0].shape
        self.output_shape = expected_output_data[0].shape

    def get_input(self):
        return self.input_shape

    def get_expected_output(self):
        return self.expected_output_data


class LayerGradientData(object):
    def __init__(self, weights, biases):
        w_dim = len(weights.shape)
        b_dim = len(biases.shape)

        assert w_dim == 3 or w_dim == 2
        assert b_dim == 2 or b_dim == 1

        self.weights_dim = w_dim
        self.biases_dim = b_dim
        self.weights_gradient = weights
        self.biases_gradient = biases
