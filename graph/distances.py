import math

def perimeter_distance(a, b, num_leaves):
    b_to_a = (abs(b - a) / num_leaves) * 2 * math.pi
    a_to_b = 2 * math.pi - b_to_a
    return min([a_to_b, b_to_a])


def hierarchical_distance(a, b):
    distance = 0
    while a != b:
        distance += 2
        a = int(a / 2)
        b = int(b / 2)
    return distance
