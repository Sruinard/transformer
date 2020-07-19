# This file has to be run in order to transform inputs to features and
# store the features in TFRecords.
# For large data workloads, we recommend to use the dataflow boilerplate
# https://bitbucket.org/ml6team/dataflow-boilerplate/src/master/

# initialize constants defined in env file
source ./scripts/env_files/constants_init.env

function error_exit() {
  # ${BASH_SOURCE[1]} is the file name of the caller.
  echo "${BASH_SOURCE[1]}: line ${BASH_LINENO[0]}: ${1:-Unknown Error.} (exit ${2:-1})" 1>&2
  exit ${2:-1}
}

while getopts :i: arg; do
  case ${arg} in
    i) INPUT_PATH="${OPTARG}";;
    \?) error_exit Unrecognized argument "${OPTARG}";;
  esac
done

[[ -n "${INPUT_PATH}" ]] || error_exit "Missing required INPUT_PATH"

# TODO activate your virtualenv here

# generate tfrecords
python trainer/transform/feature_engine.py \
    --input-path ${INPUT_PATH} \
    --feature-output-path-train ${TFRECORD_PATH_TRAIN} \
    --feature-output-path-validation ${TFRECORD_PATH_VALIDATION}
