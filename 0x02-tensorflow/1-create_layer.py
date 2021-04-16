import tensorflow.compat.v1 as tf
import tensorflow as tf

tf.disable_v2_behavior()

def create_layer(prev, n, activation):
	tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
	return tf.layers.dense(prev, n, activation=activation , name='layer')