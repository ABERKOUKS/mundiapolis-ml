#!/usr/bin/env python3
import tensorflow as tf
def create_layer(prev, n, activation):
	tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
	nn = tf.layers.dense(prev, n, activation=tf.nn.sigmoid)
	tanh= tf.layers.dense(nn, n, activation=activation , name='layer')
	return tanh
