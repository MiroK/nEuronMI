from collections import namedtuple
from itertools import dropwhile
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


def link_surfaces(model, tags, shape, links, tol=1E-10, metric=None):
    '''
    Let tags be surfaces of the model. For every surface of the shape 
    we try to pair it with one of the tagged surfaces based on metric(x, y)
    which has x as centers of all tagged surfaces and y as the surface of
    the shape

    Return a map named_surface of shape -> tag. 

    NOTE: tags is altered in the process. 
    '''
    if metric is None:
        metric = lambda x, y: np.linalg.norm(y-x, axis=1)

    tags_ = list(tags)
    # Precompute
    x = np.array([model.occ.getCenterOfMass(2, tag) for tag in tags_])

    for name in shape.surfaces:
        if name in links: continue
        y = shape.surfaces[name]

        # Correspondence
        dist = metric(x, y)
        i = np.argmin(dist)
        # Accept
        if dist[i] < tol:
            link = tags_[i]
            links[name] = link
            tags.remove(link)

    return links


def find_first(item, iterable):
    '''If iterable were indexable where would we find item'''
    pair = next(dropwhile(lambda ip: item != second(ip), enumerate(iter(iterable))))
    return first(pair)

# --------------------------------------------------------------------

if __name__ == '__main__':

    print find_first(4, range(19))
    print list(range(19)).index(4), list(range(19))
    
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
