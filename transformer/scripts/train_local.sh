#!/bin/bash
# This scripts performs local training for a TensorFlow model.
# This has to be run after genereate_features.sh is successfully executed

set -ev

echo "Training local ML model"

PACKAGE_PATH=./trainer

# initialize constants defined in env file
source ./scripts/env_files/constants_init.env

# activate your virtualenv here

# run local task
gcloud ai-platform local train \
        --module-name=trainer.task \
        --package-path=${PACKAGE_PATH} \
        -- \
        --path-to-tfrecord-train=${TFRECORD_PATH_TRAIN} \
        --path-to-tfrecord-validation=${TFRECORD_PATH_VALIDATION} \
        --serving-model-dir=${SERVING_MODEL_DIR}