'''
function that takes an upper set and returns the largest value i is such that the ith the
elementary symmetric polynomial (evaluated at the minimal elements of the upper set) is not the empty set?

elementary symmetric polynomial definition for sets:
upset={s1, s2, s3, s4} then e1 = s1 U s2...U s4, e2 = s1 X s2 U s1 X s3 U s1 X s4 U s2 X s3 ...
e4 then = s1 X s2 X s3 X s4 , so shared elements between all 4 elements
so higher degree, the less likely there are elements, because 2nd degree is just if there's shared elements between
any of pairs of sets

so the higher degree we can have means the more common elements amongst all the sets

find the highest degree we can have where the output is not the empty set, when doing it for the minimal elements

highest number of sets in minimal elements that share some element

'''

from minimalElements import findMinElems7

def maxSymPol(upset):
    minElems = findMinElems7(upset)
    allElems = minElems.union
    containDict = {}

    for i in minElems:
        for j in allElems:
            containDict[j] = []
            if i.__contains__(j):
                containDict[j].append(i)

    max = 0
    for i in range(len(allElems)):
        if containDict[allElems[i]] > max:
            max = containDict[allElems[i]]


def maxSymPol2(upset):
    minElems = findMinElems7(upset)
    allElems = unionSets(minElems)
    containDict = {elem: [] for elem in allElems}

    for s in minElems:
        for elem in s:
            containDict[elem].append(s)

    max_count = 0
    for elem, setsContaining in containDict.items():
        if len(setsContaining) > max_count:
            max_count = len(setsContaining)

    return max_count

def maxSymPol3(minElems):

    allElems = unionSets(minElems)
    containDict = {elem: [] for elem in allElems}

    for s in minElems:
        for elem in s:
            containDict[elem].append(s)

    max_count = 0
    for elem, setsContaining in containDict.items():
        if len(setsContaining) > max_count:
            max_count = len(setsContaining)

    return max_count

def unionSets(sets):
    allElements = set()
    for s in sets:
        allElements.update(s)  # Union all elements into a single set
    return allElements


upset = [{1, 2, 3}, {2, 3}, {1, 4, 9}, {5}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}]
print(maxSymPol2(upset))