# General configuration settings
DATE_FORMAT = "%Y%m%d%H%M%S"

# storage related configurations
LOG_DIR = "gs://transformer/logs"

# transform related configurations
TRAINING_PERCENTAGE = 0.8
BUFFER_SIZE = 1000

# override task related configurations with your settings
PATH_TO_TFRECORD_TRAIN = ""
PATH_TO_TFRECORD_VALIDATION = ""
NUM_EPOCHS = 1
BATCH_SIZE = 1
TRAIN_STEPS = 1
VALIDATION_STEPS = 10