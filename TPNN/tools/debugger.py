from TPNN.tools.CNNarchitecture import Layer


class Debugger(object):
    def __init__(self):
        self.mode = "on"

    def off(self):
        self.mode = "off"

    def on(self):
        self.mode = "on"

    def is_on(self):
        return self.mode == "on"

    def print_layer_derivatives(self, layer: Layer):
        if self.is_on():
            layer.print_der_config()

    def print_layer_config(self, layer: Layer):
        if self.is_on():
            layer.print_config()


def print_2d_array(arr, space):
    assert arr.shape == 2

    for y in range(arr.shape[0]):
        print(space, end='')

        for x in range(arr.shape[1]):
            print(str(arr[y][x]) + " ")
        print()
    print()


def print_block(arr, dim, dim_count, st_space):
    if dim == dim_count - 2:
        print_2d_array(arr, st_space + "  " * dim)
    else:
        space = st_space + dim * "  "
        items_count = arr.shape[0]

        for i in range(items_count):
            print(space + "[" + str(i) + "]:")
            print_block(arr[i], dim + 1, dim_count, st_space)


def print_arrays(arr, st_space):
    dim_count = len(arr.shape)

    if dim_count == 2:
        print_2d_array(arr, st_space)
    elif dim_count == 1:
        print(st_space, end='')

        for x in range(len(arr)):
            print(str(arr[x]) + " ")
        print()
    else:
        print_block(arr, 0, dim_count, st_space)