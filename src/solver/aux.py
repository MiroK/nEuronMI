from dolfin import *
import itertools
import numpy as np


def mesh_statistics(mesh_file):
    '''Get num vertices, num cells, num facets on neuron, all facets'''
    mesh, surfaces, _, _ = load_mesh(mesh_file)

    tdim = mesh.topology().dim()
    
    nvertices = mesh.topology().size_global(0)
    ncells = mesh.topology().size_global(tdim)
    # 1 2 3 21 31 are possible tags
    neuron_iterators = (SubsetIterator(surfaces, tag) for tag in (1, 2, 3, 21, 31))
    ncells_neuron = sum(1 for _ in itertools.chain(*neuron_iterators))

    nfacets = mesh.topology().size_global(tdim-1)
    return nvertices, ncells, ncells_neuron, nfacets

 
def load_mesh(mesh_file):
    '''
    The sane input is a msh file containing mesh with markers for neuron 
    domains(1) and the outside(2) and markers for surfaces (1, 2, 3) for 
    soma, axon, dendrite, (6, 5) for the bounding volume surfaces that 
    are intersected/not intersected by the probe. Probe surface has insulated 
    parts marked with (4). Probe might not be present. Moreover there 
    might optionally be surfaces tagged as 41 (then increasing monoton.
    by 1) which are conducting probe surfaces and (21 and 31) which are 
    hillocks of soma and dendrite
    '''
    comm = mpi_comm_world()
    h5 = HDF5File(comm, mesh_file, 'r')
    mesh = Mesh()
    h5.read(mesh, 'mesh', False)

    surfaces = MeshFunction('size_t', mesh, mesh.topology().dim()-1)
    h5.read(surfaces, 'facet')

    volumes = MeshFunction('size_t', mesh, mesh.topology().dim())
    h5.read(volumes, 'physical')

    comm_py = comm.tompi4py()
    # Check for presence of markers. Volume is mandatory
    local_tags = list(set(volumes.array()))
    global_tags = set(comm_py.allreduce(local_tags))
    # Neuron or Poisson
    try:
        assert global_tags == set([1, 2]), global_tags
        is_poisson = False
    except AssertionError:
        assert global_tags == set((2, )), global_tags
        is_poisson = True
        

    local_tags = list(set(surfaces.array()))
    global_tags = set(comm_py.allreduce(local_tags))

    if not is_poisson:
        # Surface, 21, 31, 41 are maybe
        # assert {1, 2, 3, 5, 6} <= global_tags, global_tags
        assert {1, 2, 3} <= global_tags, global_tags
    else:
        assert {5, 6} <= global_tags

    # Look for probe recording sites
    probe_sites = map(int, filter(lambda t: t >= 41, global_tags))
    # Check assertions
    if probe_sites:
        probe_sites = sorted(probe_sites)
        assert np.all(np.diff(probe_sites) == 1)
    else:
        assert global_tags <= {0, 1, 2, 3, 5, 6, 4, 41, 21, 31} 

    # Build the axiliary mapping which identidies the surfaces
    aux_tags = {'axon': {2, 21} & global_tags,
                'dendrite': {3, 31} & global_tags,
                'probe_surfaces': {4} & global_tags,
                'contact_surfaces': set(probe_sites)}

    return mesh, surfaces, volumes, aux_tags


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
        vertices = set(v for cell in SubsetIterator(subdomains, label) for v in cell.entities(0))
        coords = mesh.coordinates()[list(vertices)]
    return zip(coords.min(axis=0), coords.max(axis=0))


def closest_entity(x, subdomains, label):
    '''
    Return entity with smallest distance to x out of entities marked by label
    in subdomains. The distance is determined by midpoint is it's only 
    approximate.
    '''
    x = Point(*x)
    e = min(SubsetIterator(subdomains, label), key=lambda e, x=x: (x-e.midpoint()).norm())
    
    return MeshEntity(subdomains.mesh(), e.dim(), e.index())


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

    return Expression('%s < h ? A: 0' % norm_code, degree=1, **params)


def snap_to_nearest(f):
    '''An expression which evaluates f[function] at dof closest to f'''
 
    class ProxyExpression(Expression):
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

    return ProxyExpression(f, element=f.function_space().ufl_element())

# -------------------------------------------------------------------

if __name__ == '__main__':

    mesh = UnitCubeMesh(10, 10, 10)
    mesh = BoundaryMesh(mesh, 'exterior')
    V = FunctionSpace(mesh, 'CG', 2)
    f = interpolate(Expression('sin(x[0]+x[1]+x[2])', degree=2), V)
    # A proxy expression whose action at x eval f at closest point dof
    # to x
    f_ = snap_to_nearest(f)
    assert abs(f_(1., 1., 1.) - f(1., 1., 1.)) < 1E-13
    assert abs(f_(1.1, 1.1, 1.1) - f(1., 1., 1.)) < 1E-13

    # Test 2d
    mesh = UnitSquareMesh(10, 10)
    cell_f = MeshFunction('size_t', mesh, 2, 0)
    CompiledSubDomain('x[0] > 0.5 - DOLFIN_EPS && x[1] > 0.5 - DOLFIN_EPS').mark(cell_f, 1)

    assert subdomain_bbox(cell_f, 1) == [(0.5, 1.0), (0.5, 1.0)]

    # Test 3d
    mesh = UnitCubeMesh(10, 10, 10)
    cell_f = MeshFunction('size_t', mesh, 3, 0)
    CompiledSubDomain('x[0] > 0.5 - DOLFIN_EPS && x[1] > 0.5 - DOLFIN_EPS').mark(cell_f, 1)

    assert subdomain_bbox(cell_f, 1) == [(0.5, 1.0), (0.5, 1.0), (0.0, 1.0)]

    # Multi marker tests
    mesh = UnitSquareMesh(10, 10)
    cell_f = MeshFunction('size_t', mesh, 2, 0)
    CompiledSubDomain('x[0] > 0.5 - DOLFIN_EPS && x[1] > 0.5 - DOLFIN_EPS').mark(cell_f, 1)
    CompiledSubDomain('x[0] < 0.5 + DOLFIN_EPS && x[1] > 0.5 - DOLFIN_EPS').mark(cell_f, 2)
    CompiledSubDomain('x[0] < 0.5 + DOLFIN_EPS && x[1] < 0.5 + DOLFIN_EPS').mark(cell_f, 3)

    assert subdomain_bbox(cell_f) == [(0.0, 1.0), (0.0, 1.0)]
    assert subdomain_bbox(cell_f, (1, 2)) == [(0.0, 1.0), (0.5, 1.0)]
    assert subdomain_bbox(cell_f, (3, 2)) == [(0.0, 0.5), (0.0, 1.0)]
    assert subdomain_bbox(cell_f, (1, 0)) == [(0.5, 1.0), (0.0, 1.0)]
    assert subdomain_bbox(cell_f, (1, 3)) == [(0.0, 1.0), (0.0, 1.0)]

    # Closest point
    from embedding import EmbeddedMesh
    import numpy as np
    
    mesh = UnitCubeMesh(10, 10, 10)
    facet_f = MeshFunction('size_t', mesh, 2, 0)
    CompiledSubDomain('near(x[0], 0)').mark(facet_f, 1)
    CompiledSubDomain('near(x[2], 0.5)').mark(facet_f, 1)

    bmesh, subdomains = EmbeddedMesh(facet_f, [1])
    x = np.array([1, 1., 1.])
    entity = closest_entity(x, subdomains, 1)
    x0 = entity.midpoint().array()[:3]
    
    f = point_source(entity, A=Expression('3', degree=1))
    assert abs(f(x0) - 3) < 1E-15, abs(f(x0 + 1E-9*np.ones(3))) < 1E-15
