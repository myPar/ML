def print_matrix(matrix, offset):
    if len(matrix.shape) == 1:
        for i in matrix:
            print(offset + str(i), end='')
        print()

    elif len(matrix.shape) == 2:
        for row in range(len(matrix)):
            print(offset, end='')

            for column in range(len(matrix[row])):
                print(str(matrix[row][column]) + " ", end='')
            print()
    else:
        for i in range(len(matrix)):
            print(offset + "[" + str(i) + "]:")
            print_matrix(matrix[i], offset + "    ")
