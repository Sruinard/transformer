
"""
Implement your machine learning model here.
"""

import tensorflow as tf

def construct_model() -> tf.keras.Model:
    """Function that constructs and compiles a tf.keras Model

    Returns:
        tf.keras.Model: A Model object compiled with a loss function and an
        optimizer
    """

    # construct your model here
    model = tf.keras.Sequential([])

    # compile model here
    model.compile()
    return model
