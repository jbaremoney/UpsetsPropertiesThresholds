from itertools import chain, combinations

# Reuse the powerset function
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def upsetgen(S, P):
    U = set()
    P_set = set(P)  # Ensure P is a set for set operations

    def generate_upset(current, remaining):
        if not remaining:
            U.add(tuple(sorted(current)))  # Convert to tuple and add to U
            return
        for elem in list(remaining):
            new_current = current | {elem}
            new_remaining = remaining - {elem}
            generate_upset(new_current, new_remaining)  # Recurse with updated sets
            U.add(tuple(sorted(current)))  # Convert to tuple before adding

    for s in S:
        current_set = {s} if isinstance(s, str) else set(s)
        remaining_elements = P_set - current_set
        generate_upset(current_set, remaining_elements)

    return [u for u in U]  # Return the list of tuples representing the upset

def covers(S, P):
    covers = []
    Sset = set(S)
    for s in Sset:
        # check if any individual elements are covers... this part works
        for i in s:
            upset1 = upsetgen(i, P)
            print(upset1)
            if Sset.issubset(upset1):
                covers.append(tuple(i))
    return covers
        # find other possible covers

# Example usage
S = [('a',), ('b',)]
P = ('a', 'b', 'c', 'd')
covers_output = covers(S, P)
print(covers_output)
