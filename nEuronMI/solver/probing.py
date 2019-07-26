from dolfin import SubsetIterator, Point, Cell, MPI
from embedding import EmbeddedMesh
from aux import load_mesh


import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import operator

import networkx as nx
import numpy as np
import os


def probe_contact_map(path, contacts):
    '''A tag (contact) to center-point map'''
    return dict(zip(contacts, probing_locations(path, contacts)))

                    
def plot_contacts(surfaces, contacts, project=lambda x: x[1:], ax=None):
    '''
    Plot where the contacts are on the probes. Project is mapping which 
    collapsed 3d coordinates to 2d (assuming that the probe is flat). Default
    is that the probe is in plane x=value
    '''
    # Collect first triangles of contacts defined in terms of their
    # vertex index
    mesh = surfaces.mesh()
    mesh.init(surfaces.dim(), 0)
    f2v = mesh.topology()(surfaces.dim(), 0)
    
    contacts_cells = [map(f2v, map(lambda f: f.index(), SubsetIterator(surfaces, contact)))
                      for contact in contacts]
    
    x = surfaces.mesh().coordinates()
    patches = []
    centers = []
    for contact, contact_cells in zip(contacts, contacts_cells):

        patches.extend([Polygon(np.array([project(x[v]) for v in cell]))
                        for cell in contact_cells])
        # Vertices of the patch
        vertices = list(reduce(operator.or_, map(set, contact_cells)))
        centers.append(project(np.mean(x[vertices], axis=0)))
    centers = np.array(centers)

    # Heuristic for min/max coords of the plot
    xmin, ymin = np.min(centers, axis=0)
    xmax, ymax = np.max(centers, axis=0)

    xmin = xmin - (xmax-xmin)/10
    xmax = xmax + (xmax-xmin)/10

    ymin = ymin - (xmax-xmin)/10
    ymax = ymax + (xmax-xmin)/10

    if ax is None:
        fig, ax = plt.subplots()

    p = PatchCollection(patches, alpha=0.4)
    p.set_array(20*np.ones(len(patches)))
    
    ax.add_collection(p)
    
    for contact, center in zip(contacts, centers):
        ax.text(center[0], center[1], str(contact))

    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))
    ax.axis('equal')

    return ax
    

def probing_locations(path, tag):
    '''Extract probing locations of the mesh on path'''
    surfaces = load_mesh(path)[1]

    submesh, surfaces = EmbeddedMesh(surfaces, tag)  
    centers = probing_locations_for_surfaces(surfaces, tag)
    return centers


def probing_locations_for_surfaces(surfaces, tag):
    '''
    Extract probe locations as centers of tagged regions. Note that 
    if the contact surface is not flat/convex w.r.t to the domain then 
    the points might end up outside of the mesh.
    '''
    # If more
    try:
        return sum(map(lambda t: probing_locations_for_surfaces(surfaces, t),
                       tag), [])
    except TypeError:
        pass
    mesh = surfaces.mesh()
    # The actual computations
    assert isinstance(tag, (int, np.uint, np.uint32, np.uint64))
    # For we do this only from the cell function
    assert surfaces.dim() == mesh.topology().dim()
    # which represent a manifol
    assert mesh.topology().dim() == mesh.geometry().dim() - 1

    tdim = mesh.topology().dim()
    mesh.init(tdim, tdim-1)
    mesh.init(tdim-1, tdim)

    c2f, f2c = mesh.topology()(tdim, tdim-1), mesh.topology()(tdim-1, tdim)
    # The cells shall be vertices in graph with edges representing
    # facet connectivities. The we are afer connected components ...
    nodes = set(int(cell.index()) for cell in SubsetIterator(surfaces, tag))

    edges = set()
    for cell in nodes:
        # Only fully in are of interest
        for f in c2f(cell):  # At most 2
            connected_cells = set(f2c(f))
            assert len(connected_cells) in (1, 2), 'This is not a non-selfintersecting manifold'
            if len(connected_cells & nodes) == 2:  # All are in
                connected_cells.remove(cell)
                other_cell = connected_cells.pop()
                edges.add((cell, other_cell) if cell < other_cell else (other_cell, cell))
                
    graph = nx.Graph()
    graph.add_edges_from(edges)
    
    mesh.init(tdim, 0)
    c2v = mesh.topology()(tdim, 0)
    x = mesh.coordinates()
    # The idea is that connected components represent individual contact
    # surfaces. For each we grab the center of gravity
    centers = []
    for cells in nx.connected_components(graph):
        vertices = list(set(v for cell in cells for v in c2v(cell)))
        center = np.mean(x[vertices], axis=0)
        centers.append(center)
        
    return centers


class Probe(object):
    '''Perform efficient evaluation of scalar function u at fixed points'''
    def __init__(self, u, locations, t0=0.0, record=''):
        # The idea here is that u(x) means: search for cell containing x,
        # evaluate the basis functions of that element at x, restrict
        # the coef vector of u to the cell. Of these 3 steps the first
        # two don't change. So we cache them
        # Check the scalar assumption
        assert u.value_rank() == 0 and u.value_size() == 1

        # Locate each point
        mesh = u.function_space().mesh()
        limit = mesh.num_entities_global(mesh.topology().dim())
        bbox_tree = mesh.bounding_box_tree()

        cells_for_x = [None]*len(locations)
        for i, x in enumerate(locations):
            cell = bbox_tree.compute_first_entity_collision(Point(*x))
            if -1 < cell < limit:
                cells_for_x[i] = cell
        # Ignore the cells that are not in the mesh. Note that we don't
        # care if a node is found in several cells -l think CPU interface
        xs_cells = filter(lambda (xi, c): c is not None, zip(locations, cells_for_x))

        V = u.function_space()
        element = V.dolfin_element()
        coefficients = np.zeros(element.space_dimension())
        # I build a series of functions bound to right variables that
        # when called compute the value at x
        evals = []
        locations = []
        for x, ci in xs_cells:
            basis_matrix = np.zeros(element.space_dimension())

            cell = Cell(mesh, ci)
            vertex_coords, orientation = cell.get_vertex_coordinates(), cell.orientation()
            # Eval the basis once
            element.evaluate_basis_all(basis_matrix, x, vertex_coords, orientation)

            def foo(A=basis_matrix, cell=cell, vc=vertex_coords):
                # Restrict for each call using the bound cell, vc ...
                u.restrict(coefficients, element, cell, vc, cell)
                # A here is bound to the right basis_matri
                return np.dot(A, coefficients)
            
            evals.append(foo)
            locations.append(x)

        self.probes = evals
        self.locations = locations
        self.rank = MPI.rank(mesh.mpi_comm())
        self.data = []
        self.record = record
        # Make the initial record
        self.probe(t=t0)
            
    def probe(self, t):
        '''Evaluate the probes listing the time as t'''
        self.data.append([t] + [probe() for probe in self.probes])
        
        if self.record: self.save(self.record)

    def save(self, path):
        '''Dump the data to file. Each process saves its own data.'''
        # Data is one matrix with first col corresponding to time and the
        # rest is the values for probes. The locations are in the header.
        # n-th line of the header has n-th probe data
        root, ext = os.path.splitext(path)
        path = '_'.join([root, str(self.rank)]) + ext
        header = '\n'.join(['time'] + ['%r' % xi for xi in self.locations])
        np.savetxt(path, self.data, header=header)

        
# --------------------------------------------------------------------


if __name__ == '__main__':
    from dolfin import *
    mesh = UnitCubeMesh(10, 10, 10)

    if False:
        surfaces = FacetFunction('size_t', mesh, 0)
        DomainBoundary().mark(surfaces, 1)
        CompiledSubDomain('near(x[2], 1.0)').mark(surfaces, 2)
        CompiledSubDomain('near(x[2], 0.0)').mark(surfaces, 3)

        submesh, surfaces = EmbeddedMesh(surfaces, [1, 2, 3])  # To make life more difficult

        centers = probing_locations_for_surfaces(surfaces, tag=[2, 3])
        print centers

        print probing_locations('../test.h5', 41)

    V = FunctionSpace(mesh, 'CG', 1)
    f = Expression('t*(x[0]+x[1]+x[2])', t=0, degree=1)
    
    u = interpolate(f, V)
    locations = [np.array([0.2, 0.2, 0.4]),
                 np.array([0.8, 0.8, 0.2]),
                 np.array([2.0, 2.0, 2.0]),
                 np.array([0.5, 0.5, 0.1])]

    probes = Probe(u, locations, t0=0.0, record='test_record.txt')

    for t in [0.1, 0.2, 0.3, 0.4]:
        f.t = t
        u.assign(interpolate(f, V))
        probes.probe(t)

    for record in probes.data:
        t, values_x = record[0], record[1:]
        f.t = t
        for x, value_x in zip(probes.locations, values_x):
            assert abs(value_x - f(x)) < 1E-15
    
    probes.save('test.txt')
