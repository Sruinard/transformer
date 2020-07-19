"""
This file contains your logic for transforming inputs to features.
The features will be stored in TFRecords.
"""

# import packages
import argparse
import logging
import sys
from typing import List, Tuple

import tensorflow as tf

import trainer.config as cfg
import trainer.transform.feature_helpers as fh

LOGGER = logging.getLogger(__name__)

def get_args():
    """Argument parser.
    Returns:
      Dictionary of arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input-path',
        type=str,
        required=True,
        help='local or GCS location of file containing inputs')
    parser.add_argument(
        '--feature-output-path-train',
        type=str,
        required=True,
        help='local or GCS location of where to write tfrecord containing transformed features')
    parser.add_argument(
        '--feature-output-path-validation',
        type=str,
        required=True,
        help='local or GCS location of where to write tfrecord containing transformed features')
    args, _ = parser.parse_known_args()
    return args

def dataset_reader(path_to_dataset: str) -> Tuple[tf.data.Dataset]:
    """This function should contain your logic for reading the data
    and transforming it to an instance of type tf.data.Dataset

    Args:
        path_to_dataset (str): location of your stored dataset(s)

    Returns:
        Tuple[tf.data.Dataset]: a training and validation dataset containing the inputs
    """

    # Read your dataset here
    LOGGER.info("read dataset from storage location %s...", path_to_dataset)
    dataset = tf.data.experimental.CsvDataset(path_to_dataset)

    # shuffle the data
    dataset = dataset.shuffle(cfg.BUFFER_SIZE)

    # get number of samples and number of training samples
    n_samples = sum(1 for _ in dataset)
    n_training_samples = int(cfg.TRAINING_PERCENTAGE * n_samples)

    LOGGER.info("Total number of samples in dataset: %s", n_samples)
    LOGGER.info("Number of samples for training: %s samples", n_training_samples)

    # take first n_training_samples for training, remainder for validation
    train_dataset = dataset.take(n_training_samples)
    validation_dataset = dataset.skip(n_training_samples)

    return train_dataset, validation_dataset

def transform_fn(inputs: tf.Tensor, label: tf.Tensor) -> Tuple:
    """transforms inputs to features

    Args:
        text (tf.Tensor): inputs containing the text
        label (tf.Tensor): inputs containing the label

    Returns:
        Tuple: of features and label
    """
    # Your input transformations here
    features = inputs

    return features, label

def transform_fn_mapper(inputs: tf.Tensor, label: tf.Tensor) -> Tuple:
    """Allows for mapping a function containing python logic to a tf.data.Dataset

    Args:
        inputs (tf.Tensor): Tensors containing the raw input data
        label (tf.Tensor): Tensor containing the label to train on

    Returns:
        Tuple: transformed inputs (i.e. features by now) and label
    """
    # wrap in py_function for enablement of applying map function
    features, label = tf.py_function(
        transform_fn,
        inp=[inputs, label],
        Tout=(tf.int64, tf.int64)
    )
    return features, label

def serialize_example(features: List, label: int, uid: int):
    """serializes examples and label for writing to TFRecord

    Args:
        features (List):
            List containing features to be serialized
        label (int):
            target label to be serialized
        uid (int):
            ID for debugging purposes

    Returns:
        serialized tf.train.Example
    """

    # your feature spec here
    feature_spec = {
        "features": fh.int64_feature_list(value=features),
        "label": fh.int64_feature(value=label),
        "uid": fh.int64_feature(value=uid)
    }

    # convert you features to a tf.train.Example and serialize to string
    example = tf.train.Example(features=tf.train.Features(feature=feature_spec))
    return example.SerializeToString()

def tfrecord_writer(dataset: tf.data.Dataset, output_path: str):
    """writes the content of the dataset to a TFRecord dataset

    Args:
        dataset (tf.data.Dataset): dataset containing the features and label
        output_path (str): path to write the TFRecord
    """

    # create tfrecord writer object
    with tf.io.TFRecordWriter(output_path) as file_writer:
        for index, (x, y) in enumerate(dataset):
            # serialize example and write to file
            serialized_example = serialize_example(x, y, uid=index)
            file_writer.write(serialized_example)

        # pylint: disable=W0631
        LOGGER.info("%s samples written to tfrecord", index)

def main(args: argparse.ArgumentParser):
    """
    Input --> Transform --> Features --> TFRecord
    Reads inputs transform into features and writes to tfrecord

    Args:
        args (argparse.ArgumentParser): parsed arguments
    """

    # generate train and validation dataset
    train_dataset, validation_dataset = dataset_reader(args.input_path)

    # transform features on training inputs
    transformed_train_dataset = train_dataset.map(transform_fn_mapper)

    # write to training tfrecord
    LOGGER.info("Start writing to train tfrecord: %s", args.feature_output_path_train)
    tfrecord_writer(transformed_train_dataset, args.feature_output_path_train)

    # transform features on validation inputs
    transformed_validation_dataset = validation_dataset.map(transform_fn_mapper)

    # write to training tfrecord
    LOGGER.info("Start writing to validation tfrecord: %s", args.feature_output_path_validation)
    tfrecord_writer(transformed_validation_dataset, args.feature_output_path_validation)


if __name__ == "__main__":
    # set logging configuration
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(name)-4s: %(module)-4s :%(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # get parsed arguments
    feature_engine_args = get_args()

    # transform inputs to features and write to TFRecords
    main(args=feature_engine_args)
