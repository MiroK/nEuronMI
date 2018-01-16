from frechet import frechet_distance
from random import choice
import numpy as np


def lines_intersect((v0, v1), (w0, w1)):
    A = np.c_[-(v1 - v0), w1 - w0]
    b = v0 - w0

    try:
        t, s = np.linalg.solve(A, b)
        if 0 <= t <= 1 and 0 <= s <= 1:
            return True
        else:
            return False
    except np.linalg.LinAlgError:
        return False

    
def random_dir():
    return np.random.rand(2)*np.array([choice((-1, 1)), choice((-1, 1))])


def random_curve(start, length):
    curve = [start, start + random_dir()]
    curve.append(curve[-1] + random_dir())

    def intersects(p, curve):
        new_line = [curve[-1], curve[-1]+p]
        return any(lines_intersect(new_line, old_line)
                   for old_line in zip(curve[:-2], curve[1:]))

    while len(curve) < length:
        while True:
            p = random_dir()
            if not intersects(p, curve):
                curve.append(curve[-1] + p)
                break

    return np.array(curve)

        
def subpaths(path, size):
    assert len(path) >= size >= 2

    if len(path) == size: return [path]

    if size == 2: return [[path[0], path[-1]]]

    return [[path[0]] + subpath
            for i in range(1, len(path)-size+2)
            for subpath in subpaths(path[i:], size-1)]
                                                
                                                       
def nsubpaths((path, size)):
    if path == size: return 1
    if size == 2: return 1

    return sum(nsubpaths(path-i, size-1) for i in range(1, path-size+2))
# curve = np.array([[0, 0],
#                   [1, 0],
#                   [2, 0],
#                   [2, -1],
#                   [2.5, -1],
#                   [3, -1],
#                   [3, 0],
#                   [4, 0]])

n = 20
curve = random_curve(np.zeros(2), n)

subpath_length = n-3
indices = subpaths(range(n), subpath_length)

print 'Will check', len(indices), 'subpaths'
curves = (curve[i] for i in indices)

argmin_curve = min(curves, key=lambda c: frechet_distance(c, curve))

import matplotlib.pyplot as plt

plt.figure()
plt.plot(curve[:, 0], curve[:, 1], '-rx')
plt.plot(argmin_curve[:, 0], argmin_curve[:, 1], '-bo')
plt.show()

# FIXME: Subpaths must be quicker (not recursive), ideally it will be a generator
#        How many subpaths are there?
#        find_branches is not 100%
