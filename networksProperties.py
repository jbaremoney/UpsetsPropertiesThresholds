'''
Take a set of binary {-1,1} networks, let's call it A(n,l), from R to R with uniform width on the l hidden layers
(i.e. the weight matrices of our networks would be, in order, of size (nxn), (n x n), ... , (n x n), (n x n)
and our activation would be ReLU(x)=max(x,0) and all the weights would be from {-1,1}).
Pick a function f(x), a compact set [a,b] in its domain, and a real number epsilon>0
Try to write a program that actually computes the upper set of subnetworks of a given network N in A(n,l) that
approximate f(x) on [a,b] to a precision of some specified epsilon>0 (i.e. subnetworks of N such that
|N(x)-f(x)|< epsilon for all x in [a,b]).
'''
#rough draft

'''
#example of weights for net w/ 1 input node, 1 output node, 2 hidden layers of width 2.

[
    array([[ 1, -1]]),      # (1 x 2) input to first hidden layer, first node weight=1, 2nd = -1
    array([[-1,  1],        # (2 x 2) first hidden to second hidden layer, first node [-1, 1]
           [ 1, -1]]),      # 2nd node [1,-1]
    array([[ 1],            # (2 x 1) second hidden to output layer
           [-1]])
]
'''

import numpy as np
import copy

def relu(x):
    return np.maximum(0, x)
class NeuralNet: #binary default, uniform default
    def init(self, n_hidden, l, weights=[], outputSize=1, weightInit="given", mean=0, stdDev=1):
        self.n = n_hidden
        self.l = l
        self.weights = weights
        self.outputSize = outputSize
        if weightInit == "binary":
            self.weights = [np.random.choice([0, 1], (n_hidden, 1))]  # First layer weight matrix, n_hidden x 1
            self.weights += [np.random.choice([0, 1], (n_hidden, n_hidden)) for _ in range(l - 1)]  # Hidden layers
            self.weights.append(np.random.choice([0, 1], (outputSize, n_hidden)))  # Output layer, outputSize x n_hidden

        elif weightInit == "normal":
            self.weights = [np.random.normal(loc=mean, scale=stdDev, size=(n_hidden, 1))]  # input layer
            for _ in range(l - 1):  # hidden layers
                self.weights.append(np.random.normal(loc=mean, scale=stdDev, size=(n_hidden, n_hidden)))
            self.weights.append(np.random.normal(loc=mean, scale=stdDev, size=(outputSize, n_hidden))) #output

    def forward(self, x):
        output = x
        for i in range(self.l):
            output = relu(np.dot(self.weights[i], output))
        return np.dot(self.weights[-1], output)

    def isGoodNetwork(self, goalNet, epsilon, x=1):
        if abs(self.forward(x) - goalNet.forward(x)) <= epsilon:
            return True
        return False

    def getSubnetworks(self):
        shape = self.weights.shape
        numWeights = np.prod(shape)  # Total number of weights

        flatWeights = self.weights.flatten()

    def goodNetworks(self, goalNet, epsilon, x=1):
        goodNetworks = []
        for network in self.getSubnetworks():
            if network.isGoodNetwork(goalNet, epsilon, x):
                goodNetworks.append(network)
        return goodNetworks

    def contains(self, B):
        """
            Check if network A(self) contains network B.
            A contains B if all nonzero weights in B are also nonzero in A,
            but A can have additional nonzeros where B has 0s.
            """
        # iterate through corresponding weight matrices in A and B
        for weight_A, weight_B in zip(self.weights, B.weights):
            # check that A has nonzeros everywhere B has nonzeros
            if not np.all((weight_B != 0) <= (weight_A != 0)):
                return False
        return True

    def findMinimalNetworks(self, goalNet, epsilon, x=1):
        sortedUpset = sorted(self.goodNetworks(goalNet, epsilon, x), key=lambda net: sum(np.count_nonzero(w) for w in net))
        minElems = []

        while sortedUpset:
            currentNet = sortedUpset.pop(0)
            # current set is minimal. minimum size all minimal, and then all their supersets are removed
            minElems.append(currentNet)

            # if current set is a subset of anything in upperset, remove it
            sortedUpset = [s for s in sortedUpset if not s.contains(currentNet)]

        return minElems
