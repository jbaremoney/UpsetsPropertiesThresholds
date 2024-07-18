'''
Finds powerset of cartesian product of edges, weights. Iterates through powerset to find subsets that contain
f and some other pair (e, u) where (e, y) in f and u != y

This time complexity blows up (it's based on powerset), so redid in file badEdgeWeightPairs with different strategy.
'''

import itertools
import math

edges = {'e1', 'e2'}
weights = {0,1}
f = {('e1',1), ('e2', 1)}

# set of edges that are defined in f
edgesInFunc = {pair[0] for pair in f}

# set of edges that are NOT defined in f
S = edges - edgesInFunc

# N = cartesian product of N and weights but we don't really need it
lengthN = len(weights) * len(S)


# cartesian product
combos = [(edge, weight) for edge in edges for weight in weights]

def getPowerset(arr):
    return list(itertools.chain.from_iterable(itertools.combinations(arr, r) for r in range(len(arr) + 1)))

powSet = getPowerset(combos)

def getBadSets(powSet, f):
    badSets = []
    for s in powSet:
        # Convert s to set for proper subset comparison
        s_set = set(s)
        # Check if f is contained in the set
        if f.issubset(s_set):
            # Make dictionary from the subset
            subsetDict = {pair[0]: pair[1] for pair in s}
            fDict = {pair[0]: pair[1] for pair in f}
            # Flag to check if this subset is bad
            isBad = False
            for pair in s:
                x, y = pair
                # Check for a mismatch in weights for any edge in f
                if x in fDict and (x not in subsetDict or y != fDict[x]):
                    isBad = True
                    break
            if isBad:
                badSets.append(s)
    return badSets

badSets = getBadSets(powSet, f)


# n = size(f) * (size(weights) - 1) , this is the number of bad x,y pairs we can make
n = len(f) * (len(weights) - 1)

# we can call set A the set we make without using any edges that aren't in f already
# summation of (n choose i) from i=1 to n
sizeA = 0
for i in range(1, n + 1):
    sizeA += int(math.comb(n, i))

# number of new combinations for any given set in A
numNewCombs = 0
for i in range(1, lengthN + 1):
    numNewCombs += int(math.comb(lengthN, i))

totalAns = sizeA + (sizeA * numNewCombs)



print("Powerset size: " + str(len(powSet)))
print("Bad sets:" + str(badSets))

print("Formula output for set A: " + str(sizeA))
print("Formula output for total bad sets: " + str(totalAns))
print("Ratio:" + str((len(badSets)/len(powSet)).as_integer_ratio()))







