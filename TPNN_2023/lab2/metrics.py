import numpy as np
import math


def log_loss(predicted_vector, dst_vector) -> float:
    assert len(predicted_vector) == len(dst_vector)
    l = lambda x: np.vectorize(math.log2)(x)

    return -np.average(l(predicted_vector) * dst_vector)


def mse(predicted_vector, dst_vector) -> float:  # squared error
    dim = len(dst_vector)
    assert len(predicted_vector) == len(dst_vector)

    return np.sum(((predicted_vector - dst_vector) ** 2)) / dim
