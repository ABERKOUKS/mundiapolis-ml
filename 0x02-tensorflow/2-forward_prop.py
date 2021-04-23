#!/usr/bin/env python3

import tensorflow as tf

create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    
    prediction = create_layer(x, layer_sizes[0], activations[0])
    for 1 in range(len(layer_sizes)):
        prediction = create_layer(prediction, layer_sizes[i], activations[i])
    return prediction
