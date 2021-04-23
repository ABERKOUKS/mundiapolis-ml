#!/usr/bin/env python3
import tensorflow as tf
def create_layer(prev, n, activation):
	init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
	tanh = tf.layers.dense(prev, n,kernel_initializer= init, activation=activations.relu , name='layer')
	return tanh
