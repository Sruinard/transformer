#!/bin/bash

# This has to be run after train_cloud.sh is successfully executed

# initialize constants defined in env file
source ./scripts/env_files/constants_init.env

function error_exit() {
  # ${BASH_SOURCE[1]} is the file name of the caller.
  echo "${BASH_SOURCE[1]}: line ${BASH_LINENO[0]}: ${1:-Unknown Error.} (exit ${2:-1})" 1>&2
  exit ${2:-1}
}

while getopts :v:t: arg; do
  case ${arg} in
    v) MODEL_VERSION="${OPTARG}";;
    t) TIMESTAMP="${OPTARG}";;
    \?) error_exit Unrecognized argument "${OPTARG}";;
  esac
done

[[ -n "${MODEL_VERSION}" ]] || error_exit "Missing required MODEL_VERSION"
[[ -n "${TIMESTAMP}" ]] || error_exit "Missing required TIMESTAMP"

MODEL_NAME="aiplatform_model" # change to your model name, e.g. "aiplatform_model"

PYTHON_VERSION=3.7
RUNTIME_VERSION=2.1

# Deploy model to GCP using regional endpoints.
gcloud ai-platform models create "${MODEL_NAME}" --region="${REGION}"

# Deploy model version
gcloud beta ai-platform versions create ${MODEL_VERSION} \
 --model=${MODEL_NAME} \
 --region $REGION \
 --framework TENSORFLOW \
 --origin=${SERVING_MODEL_DIR}${TIMESTAMP} \
 --python-version=${PYTHON_VERSION} \
 --runtime-version=${RUNTIME_VERSION} \
 --machine-type "n1-highcpu-2"
