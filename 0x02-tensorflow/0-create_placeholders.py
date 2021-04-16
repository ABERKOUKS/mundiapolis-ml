#!/usr/bin/env python3
import tensorflow as tf
def create_placeholders(nx, classes):
	x=tf.placeholder( shape = (None,nx), dtype = tf.float32, name='x')
	y=tf.placeholder( shape = (None,classes), dtype = tf.float32, name='y')
	return x,y
