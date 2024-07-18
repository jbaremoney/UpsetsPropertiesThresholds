'''
Calculating q(F) where q(F) is the p maximum p value of all the covers p values s.t. the sum over all S in cover
of p^|S| <= 1/2
'''

import random
import math

# generate upset function because that's how we check if something covers a set
# if F is contained in the upset generated by G, G covers F

def upsetgen(S, P):
    U = set()
    P_set = set(P)

    def generate_upset(current, remaining):
        if not remaining:
            U.add(tuple(sorted(current)))  # convert to tuple and add to U
            return
        for elem in list(remaining):
            new_current = current | {elem}
            new_remaining = remaining - {elem}
            generate_upset(new_current, new_remaining)  # recurse with updated sets
            U.add(tuple(sorted(current)))  #convert to tuple before adding

    for s in S:
        current_set = {s} if isinstance(s, str) else set(s)
        remaining_elements = P_set - current_set
        generate_upset(current_set, remaining_elements)

    return [u for u in U]  # return the list of tuples representing the upset


# this is the first cover we will use, F0
def find_minimal_elements(family_of_sets):
    minimal_elements = []

    for s in family_of_sets:
        is_minimal = True

        for s_prime in family_of_sets:
            if s_prime != s and set(s_prime).issubset(set(s)):
                is_minimal = False
                break

        if is_minimal:
            minimal_elements.append(s)

    return minimal_elements



# this function performs the summation
def sum_powers_of_p(G, p):
    return sum(p ** len(S) for S in G)


# now we'll binary search for a good p value for the set
def find_p_for_half_sum(G, tolerance=.0001):
    lower, upper = 0, 1  # initial bounds
    while upper - lower > tolerance:  # continue until bounds are within tolerance
        mid = (lower + upper) / 2  # calculate midpoint
        current_sum = sum_powers_of_p(G, mid)  # calculate current sum with mid as p
        if abs(current_sum - 0.5) < tolerance:
            return mid  # if current_sum is close enough to 0.5, return mid
        elif current_sum < 0.5:
            lower = mid  # if current_sum is less than 0.5, adjust lower bound
        else:
            upper = mid  # if current_sum is greater, adjust upper bound
    return (lower + upper) / 2  # Return the midpoint of the final bounds as the best approximation

def objective_function(G):
    # find the p that makes the sum of probabilities of sets in G close to but not exceeding 1/2
    p = find_p_for_half_sum(G)
    return p  # higher p is better

def isCover(G, F, P):
    return F.issubset(upsetgen(G, P))


def generate_neighbor(G, P):
    neighbor = set(G.copy())
    chooseRemove = random.choice([True, False])

    if len(neighbor) > 1 and chooseRemove:
        # check for the amount of sets, decide if we want to remove a whole subset
        removed_subset = random.sample(list(neighbor), 1)[0]
        neighbor.remove(removed_subset)
        # print(f"Removed subset: {removed_subset}")
    elif not chooseRemove and neighbor:  # ensure neighbor is not empty
        subsetChoice = random.sample(list(neighbor), 1)[0]  # directly use the sampled subset
        subset_list = list(subsetChoice)
        if len(subset_list) > 1:  # ensure there are elements to remove
            removed_element = random.choice(subset_list)
            subset_list.remove(removed_element)
            # readd the modified subset if it's not empty
            neighbor.remove(subsetChoice)  # remove the original subset
            neighbor.add(tuple(subset_list))  # add the modified subset
            # print(f"Removed element: {removed_element} from subset: {subsetChoice}")
        else:
            pass
            # print("Subset has only one element, no further reduction possible.")
    elif len(neighbor) == 1:
        # if there's only one subset, try to remove a random element from it
        subset_tuple = next(iter(neighbor))  # extract the single subset tuple
        if len(subset_tuple) > 1:
            subset_list = list(subset_tuple)
            # remove a random element from the subset list
            removed_element = random.choice(subset_list)
            subset_list.remove(removed_element)
            # Update the neighbor
            neighbor.remove(subset_tuple)
            if subset_list:  # ensure the new subset is not empty
                neighbor.add(tuple(subset_list))
            # print(f"Removed element: {removed_element} from subset: {subset_tuple}")
    else:
        pass
        # print("G is empty, unable to modify.")

    # print("Final neighbor:", neighbor)
    return neighbor




def should_accept(current_score, neighbor_score, temperature):
    if neighbor_score > current_score:
        return True  # accept better solutions
    else:
        # accept worse solutions with a probability that decreases with temperature
        return math.exp((neighbor_score - current_score) / temperature) > random.random()

# we need to take stuff away from F0 to make look for better covers
def simulated_annealing(F0, P, initial_temperature=100, cooling_rate=0.99, tolerance=1e-5):
    current_solution = F0
    solution_score = objective_function(F0)
    current_node = F0
    current_score = objective_function(F0)
    temperature = initial_temperature
    iteration = 0  # added to track the number of iterations
    coverFails = []

    while temperature > 0.01:
        # print(f"Before generate_neighbor, Current Node: {current_node}")
        neighbor = generate_neighbor(current_node, P)

        # increment iteration at the start of the loop to ensure it's always counted
        iteration += 1
        # print(f"Iteration: {iteration}, Temperature: {temperature:.4f}")

        if not neighbor or not isCover(neighbor, F0, P):
            coverFails.append(neighbor)
            # print("Neighbor is not a valid cover or is empty.")
            if (coverFails.count(neighbor) > len(coverFails)/3) and iteration > 10:
                # need to make sure it doesn't keep trying bad stuff
                # there's definitely a better way to do this
                # print("Best set:" + str(current_solution))
                # print("Highest p:" + str(solution_score))
                return current_solution
            continue  # continue here will skip the rest of the loop if the neighbor isn't valid



        neighbor_score = objective_function(neighbor)
        # print(f"Current Score: {current_score:.4f}, Neighbor Score: {neighbor_score:.4f}")  # Check scores

        if should_accept(current_score, neighbor_score, temperature):
            # print("Neighbor accepted.")
            current_node = neighbor
            current_score = neighbor_score
            if current_score > solution_score:
                solution_score = current_score
        else:
            pass
            # print("Neighbor rejected.")

        temperature *= cooling_rate


    print("Best set:" + str(current_solution))
    print("Highest p:" + str(solution_score))
    return current_solution, solution_score


simulated_annealing(({('a', 'b'), ('b', 'c'), ('c', 'd'), ('d')}), {'a', 'b', 'c', 'd', 'e'}, initial_temperature=10, cooling_rate=0.95)

