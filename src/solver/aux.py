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
