import numpy as np
import time
import os

class RNNCell:
    def __init__(self, inp, predecessor=None, hs=None):
        self.inp = inp
        if predecessor is not None:
            self.hs = predecessor.output
            self.wi = predecessor.wi
            self.wh = predecessor.wh
            self.bias = predecessor.bias
        else:
            self.hs = hs
            self.wi = np.random.randn(self.hs.shape[0], self.inp.shape[0])
            self.wh = np.random.randn(self.hs.shape[0], self.hs.shape[0])
            self.bias = np.random.randn(self.hs.shape[0], 1)

    def print_weights(self):
        print(np.dot(self.wi,self.inp))
        print(np.dot(self.wh,self.hs))
        print(self.bias)
    
    def softmax(self, x):
        e_x = np.exp(x)
        return e_x / e_x.sum()

    def forward(self):
        self.inp_mult = np.dot(self.wi,self.inp)
        self.state_mult = np.dot(self.wh,self.hs)
        self.summation = self.inp_mult + self.state_mult
        self.output = np.tanh(self.summation)
        return self.output
    
    def backward(self, gradient, learning_rate):
        # Get gradient for tanh
        self.d_tanh = (1 - np.square(self.output)) * gradient
        # Gradient w.r.t. Wi
        self.dWi = np.dot(self.d_tanh,self.inp.T)
        self.dWh = np.dot(self.d_tanh,self.hs.T)
        self.dbias = self.d_tanh
        self.dhs = np.dot(self.wh.T,self.d_tanh)

        self.wi -= learning_rate * self.dWi
        self.wh -= learning_rate * self.dWh
        self.bias -= learning_rate * self.dbias

        return self.dhs

# hs = np.random.randn(5,1)
# inp = np.random.randn(10,1)
# rnn = RNNCell(inp,hs=hs)
# rnn.forward()
# rnn2 = RNNCell(inp,predecessor=rnn)