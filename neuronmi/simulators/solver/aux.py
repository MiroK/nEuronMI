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
        mesh.init(mesh.topology().dim(), 0)
        vertices = set(v for cell in df.SubsetIterator(subdomains, label) for v in cell.entities(0))
        coords = mesh.coordinates()[list(vertices)]
    return list(zip(coords.min(axis=0), coords.max(axis=0)))


def closest_entity(x, subdomains, label):
    '''
    Return entity with smallest distance to x out of entities marked by label
    in subdomains. The distance is determined by midpoint is it's only 
    approximate.
    '''
    x = df.Point(*x)
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
 
    class ProxyExpression(df.UserExpression):
        def __init__(self, f, **kwargs):
            super().__init__(**kwargs)
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

    return ProxyExpression(f)

# -------------------------------------------------------------------

if __name__ == '__main__':
    import dolfin as df 

    mesh = df.UnitCubeMesh(10, 10, 10)
    mesh = df.BoundaryMesh(mesh, 'exterior')
    V = df.FunctionSpace(mesh, 'CG', 2)
    f = df.interpolate(df.Expression('sin(x[0]+x[1]+x[2])', degree=2), V)

    # A proxy expression whose action at x eval f at closest point dof to x
    f_ = snap_to_nearest(f)
    assert abs(f_(1., 1., 1.) - f(1., 1., 1.)) < 1E-13
    assert abs(f_(1.1, 1.1, 1.1) - f(1., 1., 1.)) < 1E-13

    # Test 2d
    mesh = df.UnitSquareMesh(10, 10)
    cell_f = df.MeshFunction('size_t', mesh, 2, 0)
    df.CompiledSubDomain('x[0] > 0.5 - DOLFIN_EPS && x[1] > 0.5 - DOLFIN_EPS').mark(cell_f, 1)

    assert subdomain_bbox(cell_f, 1) == [(0.5, 1.0), (0.5, 1.0)]

    # Test 3d
    mesh = df.UnitCubeMesh(10, 10, 10)
    cell_f = df.MeshFunction('size_t', mesh, 3, 0)
    df.CompiledSubDomain('x[0] > 0.5 - DOLFIN_EPS && x[1] > 0.5 - DOLFIN_EPS').mark(cell_f, 1)

    assert subdomain_bbox(cell_f, 1) == [(0.5, 1.0), (0.5, 1.0), (0.0, 1.0)]

    # Multi marker tests
    mesh = df.UnitSquareMesh(10, 10)
    cell_f = df.MeshFunction('size_t', mesh, 2, 0)
    df.CompiledSubDomain('x[0] > 0.5 - DOLFIN_EPS && x[1] > 0.5 - DOLFIN_EPS').mark(cell_f, 1)
    df.CompiledSubDomain('x[0] < 0.5 + DOLFIN_EPS && x[1] > 0.5 - DOLFIN_EPS').mark(cell_f, 2)
    df.CompiledSubDomain('x[0] < 0.5 + DOLFIN_EPS && x[1] < 0.5 + DOLFIN_EPS').mark(cell_f, 3)

    assert subdomain_bbox(cell_f) == [(0.0, 1.0), (0.0, 1.0)]
    assert subdomain_bbox(cell_f, (1, 2)) == [(0.0, 1.0), (0.5, 1.0)]
    assert subdomain_bbox(cell_f, (3, 2)) == [(0.0, 0.5), (0.0, 1.0)]
    assert subdomain_bbox(cell_f, (1, 0)) == [(0.5, 1.0), (0.0, 1.0)]
    assert subdomain_bbox(cell_f, (1, 3)) == [(0.0, 1.0), (0.0, 1.0)]

    # Closest point
    from embedding import EmbeddedMesh
    import numpy as np
    
    mesh = df.UnitCubeMesh(10, 10, 10)
    facet_f = df.MeshFunction('size_t', mesh, 2, 0)
    df.CompiledSubDomain('near(x[0], 0)').mark(facet_f, 1)
    df.CompiledSubDomain('near(x[2], 0.5)').mark(facet_f, 2)

    bmesh = EmbeddedMesh(facet_f, [1, 2])
    subdomains = bmesh.marking_function

    x = np.array([1, 1., 1.])
    entity = closest_entity(x, subdomains, (1, 2))
    x0 = entity.midpoint().array()[:3]
    
    f = point_source(entity, A=df.Expression('3', degree=1))
    assert abs(f(x0) - 3) < 1E-15, abs(f(x0 + 1E-9*np.ones(3))) < 1E-15
