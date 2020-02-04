from neuronmi.simulators.solver.make_mesh_cpp import make_mesh
from neuronmi.simulators.solver.transferring import SubMeshTransfer
import dolfin as df
import numpy as np


class UnionMesh(df.Mesh):
    '''Glue together meshes'''
    def __init__(self, pieces, no_overlap=True):
        assert pieces
        assert no_overlap  # For now

        # Pieces should be the same
        tdim, = set(p.topology().dim() for p in pieces)
        gdim, = set(p.geometry().dim() for p in pieces)
        cell, = set(p.ufl_cell() for p in pieces)

        num_vertices = sum(p.num_vertices() for p in pieces)
        num_cells = sum(p.num_cells() for p in pieces)
        # Alloc space
        union_vertices = np.empty((num_vertices, gdim), dtype=float)
        union_cells = np.empty((num_cells, cell.num_vertices()), dtype='uintp')

        # There will be a mesh
        df.Mesh.__init__(self)

        vtx0, cell0 = 0, 0
        for p in pieces:
            num_vertices = p.num_vertices()
            union_vertices[vtx0:vtx0+num_vertices] = p.coordinates()

            num_cells = p.num_cells()
            cell_map = np.arange(cell0, cell0+num_cells)
            union_cells[cell_map] = p.cells() + vtx0

            vtx0 += num_vertices            
            cell0 += num_cells

            if hasattr(p, 'parent_entity_map'):
                assert self.id() not in p.parent_entity_map
                # Set the cell map
                p.parent_entity_map[self.id()] = {tdim: cell_map}
            else:
                p.parent_entity_map = {self.id(): {tdim: cell_map}}

        # Fill
        make_mesh(coordinates=union_vertices, cells=union_cells, tdim=tdim, gdim=gdim,
                  mesh=self)


def UnionFunction(meshes, fs, gmesh):
    '''Function on the global mesh with data fed by pieces'''

    elm, = set(f.function_space().ufl_element() for f in fs)

    Vg = df.FunctionSpace(gmesh, elm)

    get_from_fs = []
    # Compute mapping for getting from pieces to whole
    for mesh, f in zip(meshes, fs):
        t = SubMeshTransfer(gmesh, mesh)
        get_from_fs.append(t.compute_map(Vg, f.function_space(), strict=False))

    # The update function for global
    ug = df.Function(Vg)

    def sync(gets=get_from_fs, fs=fs, ug=ug):
        for get, f in zip(gets, fs):
            get(ug, f)
        return ug

    ug.sync = sync
    # Set
    ug.sync()

    return ug

# --------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import *

    mesh0 = RectangleMesh(Point(-2, -2), Point(-1, -1), 128, 128)
    mesh1 = RectangleMesh(Point(1, 1), Point(2, 2), 256, 256)

    mesh = UnionMesh([mesh0, mesh1])
    c = MeshFunction('size_t', mesh, 2, 0)
    c.array()[mesh0.parent_entity_map[mesh.id()][2]] = 1
    c.array()[mesh1.parent_entity_map[mesh.id()][2]] = 2

    File('foo.pvd') << c

    V0 = FunctionSpace(mesh0, 'CG', 1)
    V1 = FunctionSpace(mesh1, 'CG', 1)

    f0, f1 = Expression('A*x[0]+x[1]', degree=1, A=1), Expression('B*x[0]-2*x[1]', degree=1, B=1)
    
    fh0 = interpolate(f0, V0)
    fh1 = interpolate(f1, V1)

    ug = UnionFunction([mesh0, mesh1], [fh0, fh1], mesh)
    dx = Measure('dx', domain=mesh, subdomain_data=c)

    print sqrt(abs(assemble(inner(ug - f0, ug - f0)*dx(1))))
    print sqrt(abs(assemble(inner(ug - f1, ug - f1)*dx(2))))

    for i in range(10):
        f0.A = np.random.rand()
        f1.B = np.random.rand()

        fh0.vector()[:] = interpolate(f0, V0).vector()
        fh1.vector()[:] = interpolate(f1, V1).vector()

        print sqrt(abs(assemble(inner(ug - f0, ug - f0)*dx(1))))
        print sqrt(abs(assemble(inner(ug - f1, ug - f1)*dx(2))))

        ug.sync()
        print '>>>', sqrt(abs(assemble(inner(ug - f0, ug - f0)*dx(1))))
        print '>>>', sqrt(abs(assemble(inner(ug - f1, ug - f1)*dx(2))))
        print

        
