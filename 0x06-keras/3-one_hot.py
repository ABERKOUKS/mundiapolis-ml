#!/usr/bin/env python3

import tensorflow.keras as keras

def one_hot(labels, classes=None):
	y = keras.utils.to_categorical(labels, num_classes=num_classes)
	return y
	