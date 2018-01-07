from dolfin import Mesh, MeshEditor, MeshFunction
from dolfin import Function, FunctionSpace, dof_to_vertex_map
from itertools import count
import numpy as np


class VertexCounter(object):
    def __init__(self, zero=1E-12):
        self.vertices = np.inf*np.ones((1, 3))
        self.ticket = count()
        self.zero = zero

    def insert(self, vertex):
        # This search is faster then keeping a list and bailing out on
        # the match
        dist = np.sqrt(np.sum((self.vertices - vertex)**2, 1))
        index = np.argmin(dist)
        if dist[index] < self.zero:
            return index - 1   # Because of the initialization

        self.vertices = np.vstack([self.vertices, vertex])
        return next(self.ticket)
        

def threeD_oneD_mesh(path, round=0, center_distance=None):
    '''TODO'''
    # Sane npz file
    data = np.load(path)
    keys = ('bottoms', 'tops', 'diams', 'secs')
    assert all(key in data for key in keys)
    
    bottoms, tops, diams, secs = [data[k] for k in keys]

    bottoms = np.around(bottoms, round)
    tops = np.around(tops, round)

    n = len(bottoms)

    # In order to define the mesh the vertices have to be numbered and cell
    # defined in terms of the numbered vertices
    ZERO = 1E-12
    vc = VertexCounter(ZERO)
    cells = []

    # Store the soma - axon - dendrite as 1 2 3
    cell_data = []  
    lookup = {'soma': 1, 'axon': 2, 'dend': 3}

    # Diameters are associated with vertices which in general will not
    # be put in sequentially
    vertex_data = {}

    if center_distance is not None:
        center, distance = center_distance

        center_distance = lambda x: np.linalg.norm(x-center) < distance
    else:
        center_distance = lambda x: True
    
    for (bottom, top, sec, diam_b, diam_t) in zip(bottoms[:n-1], tops[:n-1], secs[:-1],
                                                  diams[:n-1], diams[1:n]):

        if not(center_distance(bottom) and center_distance(top)):
            continue
        
        index_b = vc.insert(bottom)
        index_t = vc.insert(top)
        
        cells.append((index_b, index_t))
        cell_data.append(lookup[sec])

        # Duplicate (but should be robust)
        vertex_data[index_b] = diam_b
        vertex_data[index_t] = diam_t

    # Remember to take away the initialization guy
    vertices = vc.vertices[1:]

    tdim = 1  # intervals
    gdim = 3  # embedded in R^3
    # Let's build the mesh
    mesh = Mesh()
    editor = MeshEditor()
    editor.open(mesh, 'interval', tdim, gdim)
    editor.init_vertices(len(vertices))
    editor.init_cells(len(cells))

    for i, v in enumerate(vertices): editor.add_vertex(i, v)

    for i, c in enumerate(cells): editor.add_cell(i, *c)

    editor.close()

    x = mesh.coordinates()
    assert np.linalg.norm(x - vertices) < ZERO
    
    # Translate the type data to cell function
    f_values = np.array(cell_data, dtype='uintp')

    f = MeshFunction('size_t', mesh, tdim, 0)
    f.set_values(f_values)
    mesh.type_info = f

    # and vertex data to vertex P1 Function
    f_values = np.zeros(len(vertex_data), dtype=float)
    for index, value in vertex_data.iteritems(): f_values[index] = value

    V = FunctionSpace(mesh, 'CG', 1)
    f = Function(V)
    f.vector().set_local(f_values[dof_to_vertex_map(V)])
    mesh.diam_info = f

    return mesh


def soma_center(path):
    # Sane npz file
    data = np.load(path)
    keys = ('tops', 'secs')
    assert all(key in data for key in keys)

    tops, secs = [data[k] for k in keys]

    return np.mean(tops[secs == 'soma'], axis=0)


def branch_terminals(mesh):
    '''
    In our definitio a branch connects two vertices of the mesh which
    are such that either a signle cell is connected to the vertex or the
    vertex is a bifurcation = 3 and more cells share it. Here we return
    a list (index) of such points and the cells that them
    '''
    assert mesh.topology().dim() == 1 and mesh.geometry().dim() > 1

    mesh.init(0, 1)
    mesh.init(1, 0)
    v2c = mesh.topology()(0, 1)

    mapping = dict()
    for v in xrange(mesh.num_vertices()):
        v_cells = v2c(v)
        if len(v_cells) == 1 or len(v_cells) > 2:
            mapping[v] = set(v_cells.tolist())

    return mapping


def find_branches(mesh):
    '''
    Produces a cell function marking each branch of the mesh with different
    color.
    '''
    terminals_map = branch_terminals(mesh)

    v2c = mesh.topology()(0, 1)
    c2v = mesh.topology()(1, 0)

    branches, terminals = [], []
    while terminals_map:
        start = vertex = next(iter(terminals_map))
        # Visited vertices of the branch
        visited = set((start, ))
        edge = terminals_map[start].pop()
        # Edges in the branch
        branch = set((edge, ))
        while True:
            v0, v1 = c2v(edge)
            vertex = v0 if v1 in visited else v1
            visited.add(vertex)
            # Terminal vertex; remove how I got here
            if vertex in terminals_map:
                # How I got here
                terminals_map[vertex].remove(edge)
                if not terminals_map[vertex]: del terminals_map[vertex]
                if not terminals_map[start]: del terminals_map[start]
                break
            # Otherwise, compute new edge
            try:
                e0, e1 = v2c(vertex)
                edge = e0 if e1 in branch else e1
            except ValueError:
                edge = v2c(vertex)[0]
            branch.add(edge)
        branches.append(branch)
        terminals.append((start, vertex))

    return terminals, branches

# --------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import File, CellFunction, SubsetIterator
    from itertools import imap
    import operator as op
    
    path = 'L5_TTPC1_cADpyr232_1_geometry.npz'

    center = soma_center(path)

    center_distance = (center, 50)
    # Load the npz file for fenics
    mesh = threeD_oneD_mesh(path, round=8, center_distance=center_distance)
    # Look for branches in the mesh that are isolated
    # These should be removed
    if center_distance is not None:
        terminals, branches = find_branches(mesh)
        # Figure out which branch is the soma
        soma_branch = set(cell.index() for cell in SubsetIterator(mesh.type_info, 1))
        
        remove_branch = set()
        for i, (v0, v1) in enumerate(terminals):
            others = map(list, terminals[:i]) + map(list, terminals[i+1:])
            others = set(sum(others, []))
            # Will remove soma branch
            if v0 not in others and v1 not in others:
                remove_branch.add(i)
        # One of them is some
        soma_branch = set([i for i in remove_branch if branches[i] == soma_branch])
        assert soma_branch
        remove_branch = remove_branch - soma_branch
        f = CellFunction('size_t', mesh, 0)
        
        for branch in remove_branch:
            for cell in branches[branch]: f[int(cell)] = 1

        File('f.pvd') << f
            
        new_vertices, new_vertex_data = [], []
        
        new_cells, new_cell_values = [], []
            
        c2v = mesh.topology()(1, 0)
        for cell in imap(op.methodcaller('index'), SubsetIterator(f, 0)):
            v0, v1 = c2v(cell)
            print (v0, v1)
            print v0, v0 in new_vertices
            try:
                v0 = new_vertices.index(v0)
            except ValueError:
                new_vertices.append(v0)
                v0 = len(new_vertices) - 1
            print v0

            print v1, v1 in new_vertices
            try:
                v1 = new_vertices.index(v1)
            except ValueError:
                new_vertices.append(v1)
                v1 = len(new_vertices) - 1
            print v1
            
            new_cells.append((v0, v1))
            new_cell_values.append(mesh.type_info[cell])
            
        # FIXME:

        x = mesh.coordinates()
        # Mesh wthout islands
        tdim = 1  # intervals
        gdim = 3  # embedded in R^3
        # Let's build the mesh
        meshr = Mesh()
        editor = MeshEditor()
        editor.open(meshr, 'interval', tdim, gdim)
        editor.init_vertices(len(new_vertices))
        editor.init_cells(len(new_cells))

        for i, v in enumerate(new_vertices):
            xi = x[v]
            editor.add_vertex(i, xi)
            new_vertex_data.append(mesh.diam_info(xi))

        for i, c in enumerate(new_cells): editor.add_cell(i, *c)

        editor.close()

        # Preserve type info
        f_values = np.array(new_cell_values, dtype='uintp')
        f = MeshFunction('size_t', meshr, tdim, 0)
        f.set_values(f_values)
        meshr.type_info = f

        # Preserve vertex info
        new_vertex_data = np.array(new_vertex_data, dtype=float)
        V = FunctionSpace(meshr, 'CG', 1)
        f = Function(V)
        f.vector().set_local(new_vertex_data[dof_to_vertex_map(V)])
        meshr.diam_info = f

        # how/doc
        # ---------
        # Simple case
        # Ellipse fit
        # The real deal
        
        File('bar.pvd') << meshr.diam_info
    # We have the geometry + data representing the radii associated with
    # vertices + data representing the type associated with each cell
    File('test.pvd') << mesh.diam_info
