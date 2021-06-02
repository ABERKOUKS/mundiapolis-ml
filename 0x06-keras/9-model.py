#!/usr/bin/env python3
"""Script to save and load a keras model"""


import tensorflow.keras as K


def save_model(network, filename):
    """Function to save a model """
    network.save(filename)
    return None


def load_model(filename):
    """ Function to load a model """
    return K.models.load_model(filename)
