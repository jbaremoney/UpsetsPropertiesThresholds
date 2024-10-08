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

class BinaryNeuralNet:

    def __init__(self, n_hidden, l, weights=[], outputSize=1):
        self.n = n_hidden
        self.l = l
        self.weights = weights

        # Weights: input layer (n_hidden, 1), hidden layers (n_hidden, n_hidden), output layer (outputSize, n_hidden)
        # not sure if this will work with binary networks, no different networks will contain each other
        # since the only way you can be greater than / contain another network is having a weight where the other
        # network has a 0 weight there.
        self.weights = [np.random.choice([0, 1], (n_hidden, 1))]  # First layer weight matrix, n_hidden x 1
        self.weights += [np.random.choice([0, 1], (n_hidden, n_hidden)) for _ in range(l - 1)]  # Hidden layers
        self.weights.append(np.random.choice([0, 1], (outputSize, n_hidden)))  # Output layer, outputSize x n_hidden


    def forward(self, x):
        output = x
        for i in range(self.l):
            output = relu(np.dot(self.weights[i], output))
        return np.dot(self.weights[-1], output)

    def generatePossWeights(self):
        possibleWeights = []

        # iterate layers
        for i in range(len(self.weights)):
            # iterate rows
            for row in range(len(self.weights[i])):
                # iterate columns
                for col in range(len(self.weights[i][row])):
                    # deep copy
                    newWeights = copy.deepcopy(self.weights)

                    if newWeights[i][row][col] == 1:
                        newWeights[i][row][col] = 0
                    else:
                        newWeights[i][row][col] = 1


                    possibleWeights.append(newWeights)

        return possibleWeights


N = BinaryNeuralNet(n_hidden=2, l=2) #given network
f = BinaryNeuralNet(n_hidden=2, l=2) #target network





# good network is one that approximates goal within epsilon
def goodNets(goalNet, given, epsilon=2):
    goal = goalNet.forward(3)  # fix .forward (no magic numbers)
    givenWeightCombs = given.generatePossWeights()
    goodNets = []

    for i in givenWeightCombs:
        currentNet = BinaryNeuralNet(n_hidden=2, l=2, weights=i)
        output = currentNet.forward(3)
        error = np.mean(np.abs(output - goal)) # mean absolute error


        if abs(error) < epsilon:
            goodNets.append(currentNet)

    return goodNets


# minimal network will be the one with most zeroed out weights ***
'''
 this is not defined quite correctly. a better definition for minimal networks as follows:
 think of all of the edges as elements in a set. then the minimal networks are the working networks that do not contain
 any other working network. and all the other sets in the upper set contain some working network.
 
 so we should store the network as a set of weights as such, so we can use the minimal elements finder already made.
 
 this can be done using matrices to represent weights
 
'''
def contains(A, B):
    """
    Check if network A contains network B.
    A contains B if all nonzero weights in B are also nonzero in A,
    but A can have additional 1s where B has 0s.
    """
    # iterate through corresponding weight matrices in A and B
    for weight_A, weight_B in zip(A, B):
        # check that A has 1's everywhere B has 1's
        if not np.all((weight_B == 1) <= (weight_A == 1)):
            return False
    return True

def findMinNets(workingNets): # from minimalElements.py
    sortedUpset = sorted(workingNets, key=lambda net: sum(np.count_nonzero(w) for w in net))

    minElems = []

    while sortedUpset:
        currentNet = sortedUpset.pop(0)
        # current set is minimal. minimum size all minimal, and then all their supersets are removed
        minElems.append(currentNet)

        # if current set is a subset of anything in upperset, remove it
        sortedUpset = [s for s in sortedUpset if not contains(s, currentNet)]

    return minElems




