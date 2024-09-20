import numpy as np

class Dense:
    def __init__(self,inp,neurons,last=False):
        self.inp = inp
        self.last = last
        self.weights = np.random.randn(neurons,self.inp.shape[0])
        self.bias = np.random.randn(neurons,1)
    
    def softmax(self,x):
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)

    def forward(self):
        self.summation = np.dot(self.weights,self.inp) + self.bias
        
        # ReLU activation
        if not self.last:
            self.output = np.maximum(0, self.summation)
        else:
            self.output = self.softmax(self.summation)

        return self.output
    
    def backward(self, gradient, learning_rate):
        if not self.last:
            self.dActivation = np.where(self.output>0,1,0) * gradient
        else:
            self.dActivation = gradient
        self.dW = np.dot(self.dActivation, self.inp.T)
        self.dB = self.dActivation
        self.dI = np.dot(self.weights.T, self.dActivation)

        self.weights -= learning_rate * self.dW
        self.dB -= learning_rate * self.dB

        return self.dI