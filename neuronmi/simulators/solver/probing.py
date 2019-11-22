from .embedding import EmbeddedMesh

from dolfin import Point, Cell, MPI
import numpy as np
import os

from dolfin import File

def get_geom_centers(surfaces, tag):
    '''
    Extract probe locations as centers of tagged regions. Note that 
    if the contact surface is not flat/convex w.r.t to the domain then 
    the points might end up outside of the mesh.
    '''
    if isinstance(tag, (int, np.uint, np.uint32, np.uint64)):
        tag = [tag]
    # If more

    mesh = surfaces.mesh()
    # For we do this only from the cell function
    if surfaces.dim() == mesh.topology().dim() - 1:
        mesh = EmbeddedMesh(surfaces, tag)
        surfaces = mesh.marking_function
    # This should be probe surface so 2d in 3d
    assert mesh.topology().dim() == mesh.geometry().dim() - 1

    # The assumption here is that each probe has a unique tag so collecting
    # cells of same tag(color) getting their unique vertices is the way to
    # get the center of masss
    tdim = mesh.topology().dim()
    mesh.init(tdim, tdim-1)
    mesh.init(tdim-1, tdim)

    c2v = mesh.topology()(tdim, 0)
    x = mesh.coordinates()
    surfaces_arr = surfaces.array()

    centers = []
    for tag_ in tag:
        tagged_cells, = np.where(surfaces_arr == tag_)
        patch_vertices = set(sum((list(c2v(c)) for c in tagged_cells), []))
        center = np.mean(x[list(patch_vertices)], axis=0)

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
        assert u.value_rank() == 0 and u.value_shape() == []

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
        xs_cells = filter(lambda xi_c : xi_c[1] is not None, zip(locations, cells_for_x))

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
            basis_matrix[:] = element.evaluate_basis_all(x, vertex_coords, orientation)

            def foo(A=basis_matrix, cell=cell, vc=vertex_coords):
                # Restrict for each call using the bound cell, vc ...
                coefficients = u.restrict(element, cell)
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
#
#
# if __name__ == '__main__':
#     from dolfin import *
#     mesh = UnitCubeMesh(10, 10, 10)
#
#     if False:
#         surfaces = FacetFunction('size_t', mesh, 0)
#         DomainBoundary().mark(surfaces, 1)
#         CompiledSubDomain('near(x[2], 1.0)').mark(surfaces, 2)
#         CompiledSubDomain('near(x[2], 0.0)').mark(surfaces, 3)
#
#         submesh, surfaces = EmbeddedMesh(surfaces, [1, 2, 3])  # To make life more difficult
#
#         centers = probing_locations_for_surfaces(surfaces, tag=[2, 3])
#         print centers
#
#         print probing_locations('../test.h5', 41)
#
#     V = FunctionSpace(mesh, 'CG', 1)
#     f = Expression('t*(x[0]+x[1]+x[2])', t=0, degree=1)
#
#     u = interpolate(f, V)
#     locations = [np.array([0.2, 0.2, 0.4]),
#                  np.array([0.8, 0.8, 0.2]),
#                  np.array([2.0, 2.0, 2.0]),
#                  np.array([0.5, 0.5, 0.1])]
#
#     probes = Probe(u, locations, t0=0.0, record='test_record.txt')
#
#     for t in [0.1, 0.2, 0.3, 0.4]:
#         f.t = t
#         u.assign(interpolate(f, V))
#         probes.probe(t)
#
#     for record in probes.data:
#         t, values_x = record[0], record[1:]
#         f.t = t
#         for x, value_x in zip(probes.locations, values_x):
#             assert abs(value_x - f(x)) < 1E-15
#
#     probes.save('test.txt')
