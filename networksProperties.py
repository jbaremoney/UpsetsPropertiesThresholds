'''
Take a set of binary {-1,1} networks, let's call it A(n,l), from R to R with uniform width on the l hidden layers
(i.e. the weight matrices of our networks would be, in order, of size (nxn), (n x n), ... , (n x n), (n x n)
and our activation would be ReLU(x)=max(x,0) and all the weights would be from {-1,1}).
Pick a function f(x), a compact set [a,b] in its domain, and a real number epsilon>0
Try to write a program that actually computes the upper set of subnetworks of a given network N in A(n,l) that
approximate f(x) on [a,b] to a precision of some specified epsilon>0 (i.e. subnetworks of N such that
|N(x)-f(x)|< epsilon for all x in [a,b]).
'''

'''
#example of weights for net w/ 1 input node, 1 output node, 2 hidden layers of width 2.

[
    array([[ 1, -1]]),      # (1 x 2) input to first hidden layer
    array([[-1,  1],        # (2 x 2) first hidden to second hidden layer
           [ 1, -1]]),
    array([[ 1],            # (2 x 1) second hidden to output layer
           [-1]])
]
'''

import numpy as np

def relu(x):
    return np.maximum(0, x)

class BinaryNeuralNet:

    def __init__(self, n, l, outputSize=1):
        self.n = n
        self.l = l

        self.weights = [np.random.choice([-1, 1], (n, n)) for _ in range(l - 1)] #hidden layer weight matrices nxn
        self.weights.insert(0, np.random.choice([-1, 1], (1, n)))  # first layer weight matrix, 1xn
        self.weights.append(np.random.choice([-1, 1], (n, outputSize)))  # output layer weight matrix, nxoutputsz


    def forward(self, x):
        output = x
        for i in range(self.l):
            output = relu(np.dot(self.weights[i], output))
        return np.dot(self.weights[-1], output)


N = BinaryNeuralNet(n=3, l=3) #given network
f = BinaryNeuralNet(n=3, l=3) #target network


print(N.weights)
#output
'''
[array([[ 1,  1, -1]]), array([[-1,  1,  1],
       [ 1, -1,  1],
       [-1,  1, -1]]), array([[ 1,  1, -1],
       [-1,  1,  1],
       [-1,  1,  1]]), array([[-1],
       [ 1],
       [-1]])]
'''






