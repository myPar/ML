

def print_line(line, max_size):
    if max_size >= len(line):
        for x in line:
            print(str(x) + " ", end='')
    else:
        st_size = max_size // 2
        end_size = max_size - st_size

        for i in range(len(st_size)):
            x = line[i]
            print(str(x) + " ", end='')

        print(" ... ", end='')

        for i in range(len(end_size)):
            x = line[len(line) - max_size + i]
            print(str(x) + " ", end='')


def print_2d_array(arr, space, max_line_size):
    assert len(arr.shape) == 2

    for y in range(arr.shape[0]):
        print(space, end='')

        print_line(arr[y], max_line_size)
        print()
    print()


def print_block(arr, dim, dim_count, st_space, max_line_size):
    if dim == dim_count - 2:
        print_2d_array(arr, st_space + "  " * dim, max_line_size)
    else:
        space = st_space + dim * "  "
        items_count = arr.shape[0]

        for i in range(items_count):
            print(space + "[" + str(i) + "]:")
            print_block(arr[i], dim + 1, dim_count, space + (dim + 1) * " ", max_line_size)


def print_arrays(arr, st_space, max_line_size):
    dim_count = len(arr.shape)

    if dim_count == 2:
        print_2d_array(arr, st_space, max_line_size)
    elif dim_count == 1:
        print(st_space, end='')

        for x in range(len(arr)):
            print(str(arr[x]) + " ", end='')
        print()
    else:
        print_block(arr, 0, dim_count, st_space, max_line_size)
