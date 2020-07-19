"""
This file contains the helper functions for converting your features
into a format as expected by tf.train.Example
"""
import tensorflow as tf

def bytes_feature(value):
    """
    creates bytes_feature for tf example
    Args:
        value: value to store in tf example

    Returns:
        feature
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def float_feature(value):
    """
    creates float_feature for tf example
    Args:
        value: value to store in tf example

    Returns:
        feature
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def int64_feature(value):
    """
    creates int64 feature for tf example
    Args:
        value: value to store in tf example

    Returns:
        feature
    """

    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def bytes_feature_list(value):
    """
    creates bytes_feature for tf example
    Args:
        value: value to store in tf example

    Returns:
        feature
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def float_feature_list(value):
    """
    creates float_feature_list for tf example
    Args:
        value: value to store in tf example

    Returns:
        feature
    """

    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def int64_feature_list(value):
    """
    creates int64 feature for tf example
    Args:
        value: value to store in tf example

    Returns:
        feature
    """

    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
