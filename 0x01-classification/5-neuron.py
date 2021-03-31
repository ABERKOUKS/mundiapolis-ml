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
 		
    def evaluate(self, X, Y):
        #A, cost = neuron.evaluate(X, Y)
        self.forward_prop(X)
        cost = self.cost(Y, self.__A)
        A = np.where(self.__A >= 0.5, 1, 0)
        return A, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        m=len(Y)
        teta= (1/m)*alpha*(np.matmul(X, (A-Y).T))
        self.__W -= teta.T 
        self.__b -= (1/m)*(alpha* np.sum(A-Y))
        