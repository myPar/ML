import math


def equals(column, value):
    if str(value) == 'nan':
        return column.isna()
    return value == column


def get_T(df_, attr, attr_value):
    if str(attr_value) == 'nan':
        return df_.loc[df_[attr].isna()]

    return df_.loc[df_[attr] == attr_value]


def get_cardinality(df_, classes):
    return df_.drop(columns=classes).drop_duplicates().shape[0]


def get_freq(classes, class_label, df_):
    # return count of rows corresponding to class_label
    return df_[equals(df_[classes[0]], class_label[0])].shape[0]


def info(df_, classes):
    classes_labels = df_[classes].drop_duplicates().to_numpy()  # get all classes labels values
    result = 0

    for label in classes_labels:
        item = get_freq(classes, label, df_) / get_cardinality(df_, classes)
        result += (item * math.log(item, 2))
    return -result


def info_x(df_, classes, attribute):
    attribute_values = df_[attribute].drop_duplicates().to_numpy()
    result = 0

    for value in attribute_values:
        df_i = get_T(df_, attribute, value) # get rows where attribute has specified value
        result += (get_cardinality(df_=df_i, classes=classes) / get_cardinality(df_=df_, classes=classes)) * info(df_i, classes)

    return result


def split(df_, attribute, classes):
    result = 0
    attribute_values = df_[attribute].drop_duplicates().to_numpy()

    for value in attribute_values:
        df_i = get_T(df_, attribute, value) # get rows where attribute has specified value
        item = get_cardinality(df_=df_i, classes=classes) / get_cardinality(df_=df_, classes=classes)
        result += (item * math.log(item, 2))

    return -result


def gain_ratio(df_, attribute, classes):
    return (info(df_, classes) - info_x(df_, classes, attribute)) / split(df_, attribute, classes)
