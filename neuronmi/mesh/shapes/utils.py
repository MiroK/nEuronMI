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


def circle_points(c, r, a=np.array([0, 0, 1.])):
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


def link_surfaces(model, tags, shape, links, tol=1E-5, metric=None, claim_last=True):
    '''
    Let tags be surfaces of the model. For every surface of the shape 
    we try to pair it with one of the tagged surfaces based on metric(x, y)
    which has x as centers of all tagged surfaces and y as the surface of
    the shape

    Return a map named_surface of shape -> tag. 

    NOTE: tags is altered in the process. 
    '''
    # Done?
    if set(shape.surfaces.keys()) <= set(links.keys()):
        return links

    if metric is None:
        metric = lambda x, y: np.linalg.norm(y-x, axis=1)
    tags_ = list(tags)
    # Precompute
    x = np.array([model.occ.getCenterOfMass(2, tag) for tag in tags_])

    for name in shape.surfaces:
        if name in links:
            continue
        y = shape.surfaces[name]
        
        # Correspondence
        dist = metric(x, y)
        i = np.argmin(dist)
        # Accept
        if dist[i] < tol:
            link = tags_[i]
            links[name] = link
            tags.remove(link)

            x = np.delete(x, i, axis=0)
            del tags_[i]

    # If there is just one remainig we claim it with warning
    if len(tags_) == 1 and claim_last:
        name, = set(shape.surfaces.keys()) - set(links.keys())
        link = tags_.pop()
        links[name] = link
        tags.remove(link)
        
    return links


def find_first(item, iterable):
    '''If iterable were indexable where would we find item'''
    pair = next(dropwhile(lambda ip: item != second(ip), enumerate(iter(iterable))))
    return first(pair)


def entity_dim(arr, dim):
    '''Get tags of entities with dim'''
    # Gmsh boolean operations result in pairs (dim, tag)
    if isinstance(arr, list):
        return list(filter(lambda item: item[0] == dim, arr))

    return sum([entity_dim(a, dim) for a in arr], [])
