"""
This is the main file illustrating a workflow for training models with TFRecords
"""
import argparse
import logging
import os
import sys
from datetime import datetime
from typing import Tuple

import tensorflow as tf

import trainer.config as cfg
from trainer.model import construct_model

LOGGER = logging.getLogger(__name__)

def get_args():
    """Argument parser.
    Returns:
      Dictionary of arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--serving-model-dir',
        type=str,
        required=True,
        help='local or GCS location serving model directory')
    parser.add_argument(
        '--path-to-tfrecord-train',
        type=str,
        default=cfg.PATH_TO_TFRECORD_TRAIN,
        help='local or GCS location containing features used for model training')
    parser.add_argument(
        '--path-to-tfrecord-validation',
        type=str,
        default=cfg.PATH_TO_TFRECORD_VALIDATION,
        help='local or GCS location containing features used for model evalutation')
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=cfg.NUM_EPOCHS,
        help='number of times to go through the data')
    parser.add_argument(
        '--batch-size',
        default=cfg.BATCH_SIZE,
        type=int,
        help='number of records to read during each training step')
    parser.add_argument(
        '--train-steps',
        default=cfg.TRAIN_STEPS,
        type=int,
        help='number of steps used for training')
    parser.add_argument(
        '--validation-steps',
        default=cfg.VALIDATION_STEPS,
        type=int,
        help='number of steps used for evaluation')
    args, _ = parser.parse_known_args()
    return args

def decode_fn(serialized_example: tf.train.Example) -> Tuple:
    """decodes a serialized example

    Args:
        serialized_example (tf.train.Example):
            a serialized tf.train.Example which will be parsed
            using the provided schema

    Returns:
        Tuple: of features and label respectively
    """

    # TODO declare your parsing schema here
    schema = {
        "features": tf.io.VarLenFeature(),
        "label": tf.io.FixedLenFeature(),
        "uid": tf.io.FixedLenFeature()
    }

    # parse example
    parsed_example = tf.io.parse_single_example(serialized_example, schema)

    # split features and label
    features = parsed_example['features']
    label = parsed_example['label']

    return features, label

# pylint: disable=C0301
def construct_datasets(input_path_train: str, input_path_validation: str, num_epochs: int, batch_size: int) -> Tuple[tf.data.Dataset]:
    """creates the training and validation_dataset datasets

    Args:
        input_path_train (str): path to training tfrecord
        input_path_validation (str): path to validation tfrecord
        num_epochs (int): num of epochs to train
        batch_size (int): number of samples per batch

    Returns:
        Tuple[tf.data.Dataset]: datasets used for training and validation
    """
    # read tfrecord and map decoder function
    train_dataset = tf.data.TFRecordDataset([input_path_train]).map(decode_fn)
    validation_dataset = tf.data.TFRecordDataset([input_path_validation]).map(decode_fn)

    # get number of samples and number of training samples
    n_training_samples = sum(1 for _ in train_dataset)
    n_validation_samples = sum(1 for _ in validation_dataset)

    LOGGER.info("Training on: %s samples", n_training_samples)
    LOGGER.info("Validating on: %s samples", n_validation_samples)

    # shuffle dataset, repeat num_epochs times and batch in batch_size
    train_dataset = train_dataset.shuffle(n_training_samples)
    train_dataset = train_dataset.repeat(num_epochs)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(5)

    # batch in batch_size
    validation_dataset = validation_dataset.batch(batch_size)
    return train_dataset, validation_dataset

def train(args):
    """
    this function contains your training logic
    """
    # Construct dataset objects
    ds_train, ds_val = construct_datasets(
        args.path_to_tfrecord_train,
        args.path_to_tfrecord_validation,
        args.num_epochs,
        args.batch_size
    )

    # Construct the model
    model = construct_model()

    # specify callbacks
    callbacks = []

    # fit the model
    model.fit(ds_train,
              epochs=args.num_epochs,
              steps_per_epoch=args.train_steps,
              validation_data=ds_val,
              validation_steps=args.validation_steps,
              verbose=2,
              callbacks=callbacks)

    # create path used for storing trained model
    serving_model_timestamp = datetime.datetime.now().strftime(cfg.DATE_FORMAT)
    serving_location = os.path.join(args.serving_model_dir, serving_model_timestamp)

    # save model
    tf.saved_model.save(model, serving_location)
    LOGGER.info("model stored at: %s", serving_location)

if __name__ == '__main__':
    # set logging configuration
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(name)-4s: %(module)-4s :%(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # parse task related arguments
    task_args = get_args()

    # run training job
    train(task_args)
