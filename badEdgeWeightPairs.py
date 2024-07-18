'''
all the supersets of f in 2^(edges x weights) that contain any pair (e,w) such that
there is some (e,u) in f with u!=w

redone more usable version of countBadEdgeWeightPairs.py
'''


def generate_bad_sets(edges, weights, f):
    # Start with an empty list for bad sets
    bad_sets = []

    # Create a dictionary from f for easy lookup
    f_dict = {edge: weight for edge, weight in f}

    # Iterate over each edge in f
    for edge, correct_weight in f:
        # Then, for each weight not equal to the correct weight for this edge, create a new set
        for w in weights:
            if w != correct_weight:
                # Create a new "bad" set by adding the (edge, w) pair to f
                new_set = f.union({(edge, w)})
                if new_set not in bad_sets:
                    bad_sets.append(new_set)

    # Now, for each edge not in f, add it with every weight to create potential new good sets
    for edge in edges - f_dict.keys():
        for w in weights:
            # Create new sets that are still "good" by adding non-conflicting pairs
            good_set = f.union({(edge, w)})
            # Extend these good sets with each bad option for the existing edges
            for existing_edge, correct_weight in f:
                for bad_weight in weights:
                    if bad_weight != correct_weight:
                        bad_set = good_set.union({(existing_edge, bad_weight)})
                        if bad_set not in bad_sets:
                            bad_sets.append(bad_set)

    return bad_sets


# Updated input with an additional edge and weight
edges = {'e1', 'e2', 'e3'}
weights = {0, 1}
f = {('e1', 0), ('e2', 0)}

bad_sets = generate_bad_sets(edges, weights, f)

print("Number of 'bad' sets:", len(bad_sets))
for b_set in bad_sets:
    print(b_set)


