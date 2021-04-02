#!/usr/bin/env python3
class Neuron(object):
    def __init__(self, nx):
        if not type(nx) is int:
  	    raise TypeError("nx must be an integer")
  	if nx < 1:
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
    
    #A, cost = neuron.train(X_train, Y_train, iterations=10)
    def train(self, X, Y, iterations=5000, alpha=0.05):
        if not type(iterations) is int:
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if not type(alpha) is float:
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")

        for it in range(iterations): #One loop
            self.forward_prop(X) #Updates the private attributes __A
            self.gradient_descent(X, Y, self.__A, alpha)  #Updates the private attributes __w, __b 
            return self.evaluate(X, Y) #Returns the evaluation of the training: A and Cost
