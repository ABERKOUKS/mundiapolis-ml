#!/usr/bin/env python3

import tensorflow as tf


def create_train_op(loss, alpha):
	train =tf.train.GradientDescentOptimizer(alpha).minimize(loss)
    return train
