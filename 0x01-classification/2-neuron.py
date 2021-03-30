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

#Generally: a = sigmoid(np.dot(w, a)+b)
    def forward_prop(self, X): 
    	z= np.matmul(self.__W, X)+ self.b #np.dot(w, a)+b:// and they asked us to use ("matmul")					
    	self.__A = 1.0/(1.0+np.exp(-z)) #The sigmoid function.
    	return self.__A
