#!/usr/bin/env python3
class Neuron(object):
    def__init__(self, nx):
        if not type(nx) is int:
            raise TypeError("nx must be an integer")
  	elif nx < 1:
            raise ValueError('nx must be a positive integer')
	    
        self.__W = np.random.normal(0, 1, (1, nx))
        self.__b = 0 # (__ to make it private)
        self.__A = 0
    
    @property
    def get_W(self):
        return self.__W #(Onlu getter of each attribute)
    @property
    def get_b(self):
	return self.__b
    @property
    def get_A(self):
        return self.__A
