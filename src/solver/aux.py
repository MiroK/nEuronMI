from dolfin import *


def load_mesh(mesh_file):
    '''
    The sane input is a msh file containing mesh with markers for neuron 
    domains(1) and the outside(2) and markers for surfaces (1, 2, 3) for 
    soma, axon, dendrite, (6, 5) for the bounding volume surfaces that 
    are intersected/not intersected by the probe. Probe surface has insulated 
    parts marked with (4). Probe might not be present. Moreover there 
    might optionally be surfaces tagged as 41 which are conducting probe 
    surfaces adn (21 and 31) which are hillocks of soma and dendrite
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
    assert global_tags == set([1, 2]), global_tags

    # Surface, 21, 31, 41 are maybe
    local_tags = list(set(surfaces.array()))
    global_tags = set(comm_py.allreduce(local_tags))
    assert {1, 2, 3, 5, 6} <= global_tags, global_tags
    assert global_tags <= {0, 1, 2, 3, 5, 6, 4, 41, 21, 31} 

    # Build the axiliary mapping which identidies the surfaces
    aux_tags = {'axon': {2, 21} & global_tags,
                'dendrite': {3, 31} & global_tags,
                'probe_surfaces': {4, 41} & global_tags}

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

# -------------------------------------------------------------------

if __name__ == '__main__':
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
