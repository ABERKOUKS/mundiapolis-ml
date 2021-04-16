def create_placeholders(nx, classes):
	x=tf.placeholder( shape = (None,nx), dtype = tf.float32)
	y=tf.placeholder( shape = (None,classes), dtype = tf.float32)
	return x,y