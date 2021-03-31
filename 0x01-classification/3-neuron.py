#!/usr/bin/env python3
class Neuron(object):
    def __init__(self, nx):
    	if not type(nx) is int:
	    raise TypeError("nx must be an integer")
  	if nx < 1:
            raise ValueError('nx must be a positive integer')
	    
	self.__W = np.random.randn(nx).reshape(1, nx)
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

    def forward_prop(self, X): #Generally: a = sigmoid(np.dot(w, a)+b)
    	z= np.matmul(self.__W, X)+ self.__b #np.dot(w, a)+b:// and they asked us to use ("matmul")					
    	self.__A = 1.0/(1.0+np.exp(-z)) #The sigmoid function.
    	return self.__A

    def cost(self, Y, A):
        m = Y.shape[1]
 	# FORWARD PROPAGATION (FROM Y TO COST)
 	#A = sigmoid(np.matmul(self.__W, Y)+ self.b)
	cost = -(1/m)*(np.sum((Y*np.log(A)) + (1-Y) *np.log(1.0000001 - A)))
 	    return cost
 		
