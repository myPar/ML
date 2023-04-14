import numpy as np


def log_loss(predicted_vector, dst_vector) -> float:
    assert len(predicted_vector) == len(dst_vector)

    return -np.sum(np.log(predicted_vector) * dst_vector)


def mse(predicted_vector, dst_vector) -> float:  # squared error
    dim = len(dst_vector)
    assert len(predicted_vector) == len(dst_vector)

    return np.sum(((predicted_vector - dst_vector) ** 2)) / dim


### classification metricx:
def get_predictions(threshold: float, predicted):   # input: 2d array of data
    result = np.zeros(predicted.shape)

    for i in range(len(predicted)):
        predicted_item = predicted[i]

        for j in range(len(predicted_item)):
            assert np.sum(predicted_item) <= 1.001

            if predicted_item[j] > threshold:
                result[i][j] = 1
                break
    return result


# calc TP or TN (regulated by indicating positive or negative label)
def calc_outcome(label, predicted_data, actual_data): # both data arrays are formatted
    assert np.sum(label) == 1 or len(label.shape) == 1
    assert predicted_data.shape == actual_data.shape
    outcome = 0

    for i in range(len(actual_data)):
        actual_item = actual_data[i]
        predicted_item = predicted_data[i]

        # if predicted item is positive and prediction is correct increment the result
        if np.array_equal(label, actual_item) and np.array_equal(label, predicted_item):
            outcome += 1

    return outcome


def calc_actual_class_count(label, actual_data):
    # calc actual positive items count
    actual_count = 0

    for i in range(len(actual_data)):
        actual_item = actual_data[i]

        if np.array_equal(label, actual_item):
            actual_count += 1

    return actual_count


def recall(positive_label, predicted_data, actual_data):
    # calc actual positive items count
    actual_positive_count = calc_actual_class_count(positive_label, actual_data)

    true_positive = calc_outcome(positive_label, predicted_data, actual_data)
    assert true_positive <= actual_positive_count

    return true_positive / actual_positive_count


def accuracy(positive_label, negative_label, predicted_data, actual_data):
    true_positive = calc_outcome(positive_label, predicted_data, actual_data)
    true_negative = calc_outcome(negative_label, predicted_data, actual_data)

    assert actual_data.shape == predicted_data.shape
    assert true_positive + true_negative <= predicted_data.shape[0]

    return (true_positive + true_negative) / predicted_data.shape[0]


def precision(positive_label, negative_label, predicted_data, actual_data):
    actual_negative_count = calc_actual_class_count(negative_label, actual_data)
    true_positive = calc_outcome(positive_label, predicted_data, actual_data)
    true_negative = calc_outcome(negative_label, predicted_data, actual_data)

    assert true_negative <= actual_negative_count

    if (true_positive + (actual_negative_count - true_negative)) == 0:
        return None

    return true_positive / (true_positive + (actual_negative_count - true_negative))
