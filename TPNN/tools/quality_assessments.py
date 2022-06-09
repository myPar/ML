import numpy as np
# quality assessments of net work

def false_positive(result_predictions, actual_predictions, target_class):
    assert len(result_predictions) == len(actual_predictions)
    result = 0

    for i in range(len(result_predictions)):
        prediction = result_predictions[i]
        actual = actual_predictions[i]

        if actual != target_class and prediction == target_class:
            result += 1

    return result


def false_negative(result_predictions, actual_predictions, target_class):
    assert len(result_predictions) == len(actual_predictions)

    result = 0

    for i in range(len(result_predictions)):
        prediction = result_predictions[i]
        actual = actual_predictions[i]

        if actual == target_class and prediction != target_class:
            result += 1

    return result


def true_negative(result_predictions, actual_predictions, target_class):
    assert len(result_predictions) == len(actual_predictions)

    result = 0

    for i in range(len(result_predictions)):
        prediction = result_predictions[i]
        actual = actual_predictions[i]

        if actual != target_class and prediction != target_class:
            result += 1

    return result


def true_positive(result_predictions, actual_predictions, target_class):
    assert len(result_predictions) == len(actual_predictions)

    result = 0

    for i in range(len(result_predictions)):
        prediction = result_predictions[i]
        actual = actual_predictions[i]

        if actual == target_class and prediction == target_class:
            result += 1

    return result


def false_positive_rate(result_predictions, actual_predictions, target_class):
    assert len(result_predictions) == len(actual_predictions)
    fp = false_positive(result_predictions, actual_predictions, target_class)
    tn = true_negative(result_predictions, actual_predictions, target_class)

    return fp / (fp + tn)


def true_positive_rate(result_predictions, actual_predictions, target_class):
    assert len(result_predictions) == len(actual_predictions)
    tp = true_positive(result_predictions, actual_predictions, target_class)
    fn = false_negative(result_predictions, actual_predictions, target_class)

    return tp / (tp + fn)


def get_ROC_points(prob_predictions, actual_predictions, target_class, get_result_predictions):
    thresholds = np.linspace(0, 1, 1000)
    points = []

    for threshold in thresholds:
        result_predictions = get_result_predictions(prob_predictions, threshold, target_class)
        fpr = false_positive_rate(result_predictions, actual_predictions, target_class)
        tpr = true_positive_rate(result_predictions, actual_predictions, target_class)
        points.append((fpr, tpr))

    return points
