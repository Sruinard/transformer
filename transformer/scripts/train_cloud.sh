#!/bin/bash

# This scripts performs cloud training for a TensorFlow model.
# This has to be run after genereate_features.sh is successfully executed

set -v

echo "Training AI Platform ML model"

DATE=$(date '+%Y%m%d_%H%M%S')

# initialize constants defined in env file
source ./scripts/env_files/constants_init.env

# JOB_NAME: the name of your job running on AI Platform.
JOB_NAME=ai_platform_job_$(date +%Y%m%d_%H%M%S)

# REGION: select a region from https://cloud.google.com/ai-platform/training/docs/regions
PYTHON_VERSION=3.7
RUNTIME_VERSION=2.1

# JOB_DIR: the output directory
JOB_DIR="gs://${BUCKET}/ai_platform_jobs/"

gcloud ai-platform jobs submit training "${JOB_NAME}" \
  --package-path trainer/ \
  --module-name trainer.task \
  --region ${REGION} \
  --python-version $PYTHON_VERSION \
  --runtime-version $RUNTIME_VERSION \
  --job-dir "${JOB_DIR}" \
  --config config.yaml \
  --stream-logs \
  -- \
  --path-to-tfrecord-train=${TFRECORD_PATH_TRAIN} \
  --path-to-tfrecord-validation=${TFRECORD_PATH_VALIDATION} \
  --serving-model-dir=${SERVING_MODEL_DIR}
