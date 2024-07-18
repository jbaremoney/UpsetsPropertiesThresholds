import numpy as np

import math

def geo_mean(iterable):
    a = np.array(iterable)
    return a.prod()**(1.0/len(a))

def sum_powers_of_p(sizes, p):
    return sum(p ** size for size in sizes)

def find_p_for_half_sum(sizes, tolerance=.0001):
    lower, upper = 0, 1  # initial bounds
    while upper - lower > tolerance:  # continue until bounds are within tolerance
        mid = (lower + upper) / 2  # calculate midpoint
        current_sum = sum_powers_of_p(sizes, mid)  # calculate current sum with mid as p
        if abs(current_sum - 0.5) < tolerance:
            return mid  # if current_sum is close enough to 0.5, return mid
        elif current_sum < 0.5:
            lower = mid  # if current_sum is less than 0.5, adjust lower bound
        else:
            upper = mid  # if current_sum is greater, adjust upper bound
    return (lower + upper) / 2  # Return the midpoint of the final bounds as the best approximation


#is it upper or lower bound
sizeArr = [1, 2, 3]
sizeArrCh = [3, 3, 3]
print(find_p_for_half_sum(sizeArr))
print(find_p_for_half_sum(sizeArrCh))








