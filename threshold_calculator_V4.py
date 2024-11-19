'''
Version 4 of threshold_calculator
November 2, 2024
Author: Martin Epstein
Based on code by Bryce Christopherson
'''

from itertools import chain, combinations
from sympy import symbols, Rational, lambdify
from sympy.polys.polyfuncs import horner
from scipy.optimize import bisect
p = symbols('p')

# For testing purposes
from time import time
from random import randrange


def CoverFunc(cover):
    '''
    Returns polynomial function for p-score of cover, minus 1/2.
    The polynomial is converted to Horner form e.g.
        x^5 + 3x^2 - 1/2 -> x^2*(x^3 + 3) - 1/2
    since this minimizes the number of operations.
    {cover} can be any iterable of iterables e.g. a set of frozen sets.
    
    What is a p-score: Suppose we select each element of the total set with probability p.
    The p-score of {cover} is the expected number of elements of {cover} whose elements are
    all selected.
    '''
    result = 0
    for c in cover:
        result += p**len(c)
    result -= Rational(1,2)
    return lambdify(p, horner(result), 'numpy')


def CriticalP(cover):
    '''
    Returns the critical probability of {cover}.
    I forget if this is the correct usage of the term "critical probability" but it makes sense to me.
    '''
    f = CoverFunc(cover)
    if f(0) > 0:
        raise ValueError("Cannot compute CriticalP for cover containing empty set")
    return bisect(CoverFunc(cover), 0, 1)


def NontrivialSubsets(my_set, size=0):
    '''
    If size = 0 or not given, yield all subsets of at least 2 elements.
    If size > 0 yield all subsets of exactly {size} elements.
    '''
    s = list(my_set)
    if size == 0:
        return chain.from_iterable(combinations(s,r) for r in range(2, len(s) + 1))
    return combinations(s, size)


def SelfIntersection(S):
    '''
    When S is a nonempty set of frozensets, return the intersection of
    these frozensets.
    '''
    S = set(S)
    output = S.pop()
    while S:
        output = output.intersection(S.pop())
    return output

def SetMinus(G, S):
    '''
    Equivalent to set(G) - set(S)
    '''
    return {g for g in G if all(g != s for s in S)}

def SubsetReplace(generators, S):
    '''
    When S is a subset of generators and SelfIntersection(S) is nonempty,
    remove the elements of S and add intersection(S).
    Return None if intersection(S) is empty
    '''    
    int_S = SelfIntersection(S)
    if not int_S:
        return None
    output = SetMinus(generators, S)
    output.add(int_S)
    return output

def ReduceGeneratorSet1(generators):
    '''
    The idea here is to use SubsetReplace on various subsets to try to
    increase the critical probability.
    This version of ReduceGeneratorSet tries all subsets of size 2 or greater
    and picks the best subset.
    This version is much slower than the subsequent versions, but I'm more
    confident that it always yields q(F). That said, I haven't found an example where
    any of the versions disagree.
    '''
    best_G = generators
    best_p = CriticalP(generators)
    found_better = False
    
    for S in NontrivialSubsets(generators):
        G_S = SubsetReplace(generators, set(S))
        if G_S:  # Pass if subsetReplace returns None i.e. if selfIntersection(S) is empty
            p_S = CriticalP(G_S)
            if p_S > best_p:
                best_p = p_S
                best_G = G_S
                found_better = True
    
    return best_p, best_G, found_better


def ReduceGeneratorSet2(generators):
    '''
    This version of ReduceGeneratorSet only tries subsets of size exactly 2
    and goes with the best such subset.
    Took 24 seconds to compute q for 40 random generators, total set of size 10.
    '''
    best_G = generators
    best_p = CriticalP(generators)
    found_better = False
    
    for S in NontrivialSubsets(generators, 2):
        G_S = SubsetReplace(generators, set(S))
        if G_S:  # Pass if subsetReplace returns None i.e. if selfIntersection(S) is empty
            p_S = CriticalP(G_S)
            if p_S > best_p:
                best_p = p_S
                best_G = G_S
                found_better = True
    
    return best_p, best_G, found_better


def ReduceGeneratorSet3(generators):
    '''
    This version of ReduceGeneratorSet only tries subsets of size exactly 2
    and goes with the first improvement rather than the best improvement.
    Took 4.5 seconds to compute q for 40 random generators, total set of size 10.
    '''
    p = CriticalP(generators)
    
    for S in NontrivialSubsets(generators, 2):
        G_S = SubsetReplace(generators, set(S))
        if G_S:  # Pass if subsetReplace returns None i.e. if selfIntersection(S) is empty
            p_S = CriticalP(G_S)
            if p_S > p:
                return p_S, G_S, True

    return p, generators, False


def ReduceGeneratorSet4(generators):
    '''
    This is the same as version 3, but instead of computing CriticalP(G_S) for every S
    we just plug p into the cover function and see if the result is < 0.

    '''
    p = CriticalP(generators)
    
    for S in NontrivialSubsets(generators, 2):
        G_S = SubsetReplace(generators, set(S))
        if G_S:  # Pass if subsetReplace returns None i.e. if selfIntersection(S) is empty
            if CoverFunc(G_S)(p) < 0:
                return CriticalP(G_S), G_S, True

    return p, generators, False


def q1(generators):
    best_p, best_G, found_better = ReduceGeneratorSet1(generators)

    while found_better:
        best_p, best_G, found_better = ReduceGeneratorSet1(best_G)
    
    return best_p, best_G


def q2(generators):
    '''
    Took 76.7 seconds to compute q for 50 generators, |total set| = 20
    '''
    best_p, best_G, found_better = ReduceGeneratorSet2(generators)

    while found_better:
        best_p, best_G, found_better = ReduceGeneratorSet2(best_G)
    
    return best_p, best_G


def q3(generators):
    '''
    Took 29.4 seconds to compute q for 50 generators, |total set| = 20
    '''
    best_p, best_G, found_better = ReduceGeneratorSet3(generators)

    while found_better:
        best_p, best_G, found_better = ReduceGeneratorSet3(best_G)
    
    return best_p, best_G


def q4(generators):
    '''
    Took 19.3 seconds to compute q for 50 generators, |total set| = 20
    '''
    best_p, best_G, found_better = ReduceGeneratorSet4(generators)

    while found_better:
        best_p, best_G, found_better = ReduceGeneratorSet4(best_G)
    
    return best_p, best_G



# Testing

generators = set()

n = 20
count = 50
for _ in range(count):
    new_gen = set()
    for i in range(n):
        if randrange(2) == 1:
            new_gen.add(i)
    if new_gen:
        generators.add(frozenset(new_gen))

# If there are relatively few generators then this often equals q(F)
print("Critical probability of generator set:")
print(f"    {CriticalP(generators)}")

'''
These tests are commented out because q1 and q2 are too slow.

start = time()
print("q the first way:")
print(f"    {q1(generators)[0]}")
end = time()
print(f"    time: {end - start}")
'''

start = time()
print("q the second way:")
print(f"    {q2(generators)[0]}")
end = time()
print(f"    time: {end - start}")

start = time()
print("q the third way:")
print(f"    {q3(generators)[0]}")
end = time()
print(f"    time: {end - start}")

start = time()
print("q the fourth way:")
print(f"    {q4(generators)[0]}")
end = time()
print(f"    time: {end - start}")