import numpy as np

# activation functions
def ident(x):
    return x


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def th(x):
    return np.tanh(x)


# applied to 1-d array
def ELU(x, alpha):
    for i in range(len(x)):
        if x[i] < 0:
            x[i] = alpha * (np.exp(x) - 1)
    return x


def RELU(x):
    return np.maximum(x, 0)


# derivatives of activation functions
def ident_der(x):
    return 1


def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x))


def th_der(x):
    return 1 - th(x) ** 2


def ELU_der(x, alpha):
    for i in range(len(x)):
        if x >= 0:
            x[i] = 1
        else:
            x[i] = alpha * np.exp(x)
    return x


def RELU_der(x):
    if np.isscalar(x):
        if x >= 0:
            return 1
        else:
            return 0
    else:
        x[x >= 0] = 1
        x[x < 0] = 0

    return x


def get_der(function_ident):
    if function_ident == sigmoid:
        return sigmoid_der
    elif function_ident == ident:
        return ident_der
    elif function_ident == th:
        return th_der
    elif function_ident == ELU:
        return ELU_der
    elif function_ident == RELU:
        return RELU_der
    else:
        assert False


def softmax(arr):
    numerator = np.exp(arr)

    return numerator / np.sum(numerator)


# --------- loss functions:

# checks is vector an one-hot-encoding vector
def check_one_hot_encoding(one_hot_enc_vector):
    assert np.max(one_hot_enc_vector) == 1 and np.sum(one_hot_enc_vector) == 1


# get idx od 1 in one-hot-encoding vector
def get_ones_pos(one_hot_enc_vector):
    check_one_hot_encoding(one_hot_enc_vector)

    for i in range(len(one_hot_enc_vector)):
        if one_hot_enc_vector[i] == 1:
            return i

    assert False


# logarithmic loss function
def log_loss(one_hot_enc_vector, result_vector):
    assert len(one_hot_enc_vector.shape) == len(result_vector.shape) == 1
    assert one_hot_enc_vector.shape == result_vector.shape
    check_one_hot_encoding(one_hot_enc_vector)

    return -np.sum(np.log(result_vector) * one_hot_enc_vector)


# gets loss function gradient vector with respect to predicted vector
def der_loss_y(loss_function, predicted_vector, actual_vector):
    assert predicted_vector.shape == actual_vector.shape
    assert len(predicted_vector.shape) == len(actual_vector.shape) == 1

    result = np.zeros(actual_vector.shape)

    # gets gradient vector for each type of loss function
    if loss_function == log_loss:
        idx = get_ones_pos(actual_vector)
        result[idx] = 1 / predicted_vector[idx]

        return result
    else:
        assert False


def get_value(softmax_result):  # returns predicted value from softmax result or one-hot-enc vector
    assert 0.95 <= np.sum(softmax_result) <= 1.1

    return np.argmax(softmax_result)


# return corresponding one hot encoding vector to number
def get_one_hot_encoding(number: int):
    assert 9 >= number >= 0
    result = np.zeros((10, ))
    result[number] = 1

    return result


# prob_prediction - prob vector, target_class - value, threshold - value
def get_prediction(prob_prediction, target_class, threshold):
    assert len(prob_prediction) == 10 and np.max(prob_prediction) <= 1 and np.min(prob_prediction) >= 0
    assert 9 >= target_class >= 0
    assert 1 >= threshold >= 0

    if prob_prediction[target_class] >= threshold:
        return int(target_class)
    else:
        return -1


# prob_predictions - prob vectors, target_class - value, threshold - value
def get_predictions(prob_predictions, target_class, threshold):
    pred_count = len(prob_predictions)
    result_predictions = np.zeros((pred_count,)) - 1

    for i in range(pred_count):
        result_predictions[i] = get_prediction(prob_predictions[i], target_class, threshold)

    return result_predictions


def average_loss(actual_data, predicted_data, loss_function):
    return np.mean([loss_function(actual_data[i], predicted_data[i]) for i in range(len(actual_data))])


# act_data - one-hot-enc vector, pred_data - softmax result vector
def categorical_accuracy(actual_data, predicted_data):
    count_true_predicted = 0
    count_total = len(actual_data)

    for i in range(len(actual_data)):
        act_item = actual_data[i]
        pred_item = predicted_data[i]

        if get_value(act_item) == get_value(pred_item):
            count_true_predicted += 1

    return count_true_predicted / count_total
