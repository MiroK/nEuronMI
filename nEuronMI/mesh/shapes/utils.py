from collections import namedtuple
import numpy as np


def as_namedtuple(d):
    '''Dictionary -> namedtuple instance'''
    keys = list(d.keys())
    foo = namedtuple('foo', keys)
    return foo(**d)


def has_positive_values(d, keys=None):
    '''Dict of positive numbers [selection of keys]'''
    if keys is None:
        return all(v > 0 for v in d.values())

    return all(d[k] > 0 for k in d if k in keys)


def unit_vector(n):
    '''|n| = 1'''
    return n/np.linalg.norm(n)


def circle_points(c, r, a=np.array([0, 0, 1])):
    '''Four points on radius r circle in plane containing c with normal a'''
    assert len(c) == len(a) == 3
    
    n = unit_vector(a)
    lmbda, U = np.linalg.eigh(np.eye(3) - np.outer(n, n))
    assert abs(lmbda[0]) < 1E-14, 'This is not projection'

    # Extract basis vectors of the plane
    u, v = U.T[1:]

    return np.array([c+r*u, c-r*u, c+r*v, c-r*v])


def first(iterable):
    '''A better [0]'''
    return next(iter(iterable))

def second(iterable):
    '''A better [1]'''
    it = iter(iterable)
    next(it)
    return first(it)

# --------------------------------------------------------------------

if __name__ == '__main__':
    d = {'a': 1, 'b': 2}
    print(as_namedtuple(d))


    n = np.array([1, 1, 1])
    n = n/np.linalg.norm(n)
    c = np.array([0, 0, 0])
    r = 0.24
    pts = circle_points(c, r, a=n)
    # point - center is perp to axis; distance is correct
    for p in pts:
        print np.dot(p-c, n)
        print '  ', np.linalg.norm(p-c)

    print first((0, 1, 2))
    print second((1, 2))
