#!/usr/bin/env python3

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs, validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """
    Function to train a model using keras and validate data
    Args
    Returns: History object generated after training the model

    """
    callback_ES = []
    ES = K.callbacks.EarlyStopping(monitor='val_loss', mode='min',
                                   patience=patience)
    if validation_data and early_stopping:
        callback_ES.append(ES)

    history = network.fit(x=data, y=labels, batch_size=batch_size,
                          epochs=epochs, validation_data=validation_data,
                          callbacks=callback_ES,
                          verbose=verbose, shuffle=shuffle,)
    return history
