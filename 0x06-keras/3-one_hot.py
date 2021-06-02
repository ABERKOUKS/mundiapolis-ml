#!/usr/bin/env python3

import tensorflow.keras as K


def one_hot(labels, classes=None):

    One_hot = K.utils.to_categorical(labels, num_classes=classes)
    return One_hot
    
