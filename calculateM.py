'''
check if dim(F) = |F0| + 1 - max{1 <= m <= |F0|: symPolyM(Si:Si in F0) != empty set}


F0 = {{a, b, c}, {a, b, d}, {e}, {f, g}}
then max sym poly will be 2
so F0 covered by {a}, {e}, {f, g}, because a is most shared i for i in S in F0
m = 4 + 1 -2 = 3 so it works here

but if i have a property where there are more shared elements (not just the most shared one)
like F0 = {{a, b, c}, {a, d,e},{a, k} , {f, g}, {f, c}}, a still most shared, but could still shrink more by covering
stuff that shares f
the max sym poly will still be 2
so we have 4 + 1 - 2 = 3,
but this F0 could be covered by {a}, {f} so m = 2 here.

a way to calculate m would be by using similar structure to the elemSymPoly program for getting the most shared
i for i in S in F0, we could recursively apply the formula shown in the paper to the minimal elements, subtracting
the sets that all share the max shared i each time


'''

def unionSets(sets):
    allElements = set()
    for s in sets:
        allElements.update(s)  # Union all elements into a single set
    return allElements

def getM(minElems):
    allElems = unionSets(minElems)
    containDict = {elem: [] for elem in allElems}

    for s in minElems:
        for elem in s:
            containDict[elem].append(s)


    while len(minElems) > 0:
        # remove each set in F0 containing the most shared i for i in S in F0, add the {i} covering them
        # repeat
        max_count = 0
        for I, setsContaining in containDict.items():
            if len(setsContaining) > max_count:
                max_count = len(setsContaining)
                max_I = I

        for set in containDict[max_I]:
            minElems.remove(set)
            minElems.append(frozenset(max_I))
        containDict.pop(max_I)

    return len(minElems)