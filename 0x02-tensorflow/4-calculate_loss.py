#!/usr/bin/env python3

import tensorflow as tf


def calculate_loss(y, y_pred):
	loss= tf.compat.v1.losses.softmax_cross_entropy(y, y_pred)
    return loss
