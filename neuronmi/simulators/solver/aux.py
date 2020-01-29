import dolfin as df
import itertools
import numpy as np


def as_tuple(maybe):
    '''Tuple of numbers'''
    if isinstance(maybe, (int, float)):
        return (maybe, )
    return tuple(maybe)


def subdomain_bbox(subdomains, label=None):
    '''
    Draw a bounding box around subdomain defined by entities in `subdomains`
    tagged with label. Return a d-tuple of intervals such that their 
    cartesion product forms the bounding box.
    '''
    if hasattr(label, '__iter__'):
        return [(min(I[0] for I in intervals), max(I[1] for I in intervals))
                for intervals in zip(*(subdomain_bbox(subdomains, l) for l in label))]
    
    mesh = subdomains.mesh()
    if label is None:
        coords = mesh.coordinates()
    else:
        arr = subdomains.array()
        
        mesh.init(mesh.topology().dim(), 0)
        c2v = mesh.topology()(mesh.topology().dim(), 0)
        vertices = set(np.concatenate(list(map(c2v, np.where(arr == label)[0]))))
        coords = mesh.coordinates()[list(vertices)]
    return list(zip(coords.min(axis=0), coords.max(axis=0)))


def closest_entity(x, subdomains, label=None):
    '''
    Return entity with smallest distance to x out of entities marked by label
    in subdomains. The distance is determined by midpoint is it's only 
    approximate.
    '''
    x = df.Point(*x)
    # Grab all tags
    if label is None:
        label = set(subdomains.array())
    label = as_tuple(label)
    sub_iter = itertools.chain(*[df.SubsetIterator(subdomains, l) for l in label])

    e = min(sub_iter, key=lambda e, x=x: (x-e.midpoint()).norm())

    return df.MeshEntity(subdomains.mesh(), e.dim(), e.index())


def point_source(e, A, h=1E-10):
    '''
    Create a point source (h cutoff) with amplitude A at the entity center
    '''
    gdim = e.mesh().geometry().dim()
    x = e.midpoint().array()[:gdim]
    
    degree = A.ufl_element().degree()

    norm_code = '+'.join(['pow(x[%d]-x%d, 2)' % (i, i) for i in range(gdim)])
    norm_code = 'sqrt(%s)' % norm_code

    params = {'h': h, 'A': A}
    params.update({('x%d' % i): x[i] for i in range(gdim)})

    return df.Expression('%s < h ? A: 0' % norm_code, degree=1, **params)


def snap_to_nearest(f):
    '''An expression which evaluates f[function] at dof closest to f'''
 
    class ProxyExpression(df.Expression):
        def __init__(self, f, **kwargs):
            self.f = f
            V = f.function_space()
            self.y = V.tabulate_dof_coordinates().reshape((V.dim(), -1))
            self.snaps = {}
            
        def eval(self, value, x):
            x = self.snap_to_nearest(x)
            value[:] = self.f(x)
            # Keep track of where the evaluation happend
            self.eval_point = x

        def value_shape(self):
            return f.ufl_element().value_shape()

        def snap_to_nearest(self, x):
            x = tuple(x)
            out = self.snaps.get(x, None)
            # Memoize
            if out is None:
                out = self.y[np.argmin(np.sqrt(np.sum((self.y - x)**2, axis=1)))]
                self.snaps[x] = out
                
            return out

    return ProxyExpression(f, degree=f.function_space().ufl_element().degree())


class SiteCurrent(df.Expression):
    '''normal*I where I can vary in time and normal is fixed'''
    def __init__(self, I, n, **kwargs):
        self.n = n
        self.I = I
        self._time = 0
        self.t = 0

    def value_shape(self):
        return (3, )

    def eval(self, values, x):
        values[:] = self.n*self.I(x)

    @property
    def t(self):
        return self._time

    @t.setter
    def t(self, t):
        self._time = t
        hasattr(self.I, 't') and setattr(self.I, 't', self._time)


def surface_normal(tag, facet_f, point):
    '''Normal of taged surface which points away from the point'''
    # NOTE: as this is a constant it will only work for a flat surface
    mesh = facet_f.mesh()
    tdim = facet_f.dim()
    assert tdim == mesh.topology().dim()-1
    assert mesh.geometry().dim() == 3

    point = df.Point(*point)
    facets,  = np.where(facet_f.array() == tag)
    facets = iter(facets)
    
    first = df.Facet(mesh, next(facets))
    n = first.normal()
    # Is this a flat surface
    assert all(abs(abs(df.Facet(mesh, f).normal().dot(n))-1) < 1E-10 for f in facets)

    mid = first.midpoint()

    return n.array() if n.dot(mid-point) > 0 else -1*n.array()

