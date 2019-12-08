#!/usr/bin/env python

import numpy as np
from abc import abstractmethod


def flatten(input_value):
    return input_value.reshape(input_value.shape[0], -1)

def normalize(input_value):
    res = input_value - np.min(input_value)
    return res / np.max(res)

def image_preprocess(image_data):
    return normalize(flatten(image_data))


# TODO: fix abstraction
class ActivationFunction:    
    @abstractmethod
    def activate(self, x):
        pass

    @abstractmethod
    def derivative(self, x):
        pass
    
    def __call__(self, x):
        return self.activate(x)

    
# And here some sample activation functions
class Sigmoid(ActivationFunction):
    def activate(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def derivative(self, x):
        return self.activate(x) * (1.0 - self.activate(x)) 

class ReLU(ActivationFunction):
    def activate(self, x):
        return x * (x > 0)

    def derivative(self, x):
        return 1.0 * (x > 0)
    

class tanh(ActivationFunction):
    def activate(self, x):
        return np.tanh(x)

    def derivative(self, x):
        return 1.0 / np.cosh(x) ** 2


ACTIVATION_FUNCTIONS = {
    'sigmoid': Sigmoid(),
    'relu': ReLU(),
    'tanh': tanh(),
}

def get_activation_function(actv_func):
    if isinstance(actv_func, str):
        if actv_func not in ACTIVATION_FUNCTIONS:
            raise Exception('activation "%s" not found' % actv_func)
        actv_func = ACTIVATION_FUNCTIONS[actv_func]
    return actv_func


class Layer:
    def __init__(self, n, prev_n, actv_func):
        self.actv_func = get_activation_function(actv_func)
        self.n = n
        self.prev_n = prev_n
        self.initialize()
        
    def initialize(self):
        self.w = np.random.uniform(low=0, high=+1, size=(self.n, self.prev_n))
        self.b = np.random.uniform(low=0, high=+1, size=(self.n, 1))

        # These parameters will be used in backprop
        self.x0 = 0
        self.z0 = 0
        self.dw = 0
        self.db = 0
        
        # Debug plot
        self.hist_w = []
        self.hist_b = []
    
    def set_params(self, new_w, new_b, new_func=None):
        if new_w.shape != self.w.shape:
            raise Exception('weight size mismatch. Expecting %s but got %s' % (self.w.shape, new_w.shape))
        if new_b.shape != self.b.shape:
            raise Exception('bias size mismatch. Expecting %s but got %s' % (self.b.shape, new_b.shape))
            
        self.w = new_w
        self.b = new_b
        if new_func is not None:
            self.actv_func = get_activation_function(new_func)
    
    def forward(self, x):
        z = self.w.dot(x) + self.b
        a = self.actv_func(z)
        
        self.z0 = z
        self.x0 = x
        
        return a
    
    def backward(self, error, m):
        delta = error * self.actv_func.derivative(self.z0)
        self.dw = delta.dot(self.x0.T) / float(m)
        self.db = delta.dot(np.ones((m,1))) / float(m)
        return self.w.T.dot(delta)
    
    def optimize_weights(self, eta):
        self.w += eta * self.dw
        self.b += eta * self.db
        
        self.hist_w.append(self.w.flatten())
        self.hist_b.append(self.b.flatten())
        

class Network:
    def __init__(self, input_size):
        self.layers = []
        self.last_layer_size = input_size
        self.lr = 0.01
        self.initialize()
    
    def add_layer(self, n, activation='sigmoid'):
        self.layers.append(Layer(
            n,
            self.last_layer_size,
            activation
        ))
        self.last_layer_size = n
    
    def predict(self, x0):
        z = x0
        for l in self.layers:
            z = l.forward(z)
        return z
    
    def backpropagate(self, x0, y0):
        m = x0.shape[1]
        y_hat = self.predict(x0)
        error = y0-y_hat

        for i in reversed(range(len(self.layers))):
            error = self.layers[i].backward(error, m)
        
        for i in range(len(self.layers)):
            self.layers[i].optimize_weights(self.lr)

    def initialize(self):
        self.loss_history = []
        for l in self.layers:
            l.initialize()
        
    def train(self, x, y, batch_size, epochs, lr=None, initialize=False):
        if initialize:
            self.initialize()
        
        if lr is not None:
            self.lr = lr
        
        for e in range(epochs): 
            i=0
            batch_loss = []
            while(i<x.shape[1]):
                x_batch = x[:, i:i+batch_size]
                y_batch = y[:, i:i+batch_size]
                i += batch_size
                
                self.backpropagate(x_batch, y_batch)
                batch_loss.append(np.linalg.norm(self.predict(x_batch) - y_batch))

            self.loss_history.append(np.mean(batch_loss))
            
    def plot_loss(self, weight_history=False):
        with plt.xkcd():
            plt.plot(nn.loss_history)
            plt.title('loss')
            plt.show()

            for i, l in enumerate(self.layers):
                plt.plot(l.hist_w)
                plt.title('W%d' % (i+1))
                plt.show()

                plt.plot(l.hist_b)
                plt.title('B%d' % (i+1))
                plt.show()
