#!/usr/bin/env python3
import tensorflow as tf
def create_layer(prev, n, activation):
	tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
	return tf.layers.dense(prev, n, activation=activation , name='layer')
