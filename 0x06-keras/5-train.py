#!/usr/bin/env python3
"""Script to train a model using keras"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs, validation_data=None, verbose=True, shuffle=False):

    if validation_data:
        validation_data = validation_data
    else:
        validation_data = None

    return network.fit(x=data, y=labels, batch_size=batch_size,
                       epochs=epochs,
                       validation_data=validation_data,
                       verbose=verbose,
                       shuffle=shuffle)
