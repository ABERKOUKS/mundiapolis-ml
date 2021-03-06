#!/user/bin/env python3
class Neuron(object):
	def __init__(self, nx):
		if not type(nx) is int:
			raise TypeError("nx must be an integer")

		if nx<1:
			raise ValueError("nx must be a positive integer")
		self.W= np.random.normal(0,1,(1, nx))
		self.b=0
		self.A=0
