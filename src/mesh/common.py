from dolfin import Mesh, MeshEditor
from itertools import count
import numpy as np


class VertexCounter(object):
    '''
    Keep track of unique vertices by their index. Basically
    try:
        return seen.index(vertex)
    except ValueError:
        seen.append(vertex)
        return seen.index(vertex)
    '''
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

    
def branch_terminals(mesh):
    '''
    In our definition a branch connects two vertices of the mesh which
    are such that either a signle cell is connected to the vertex or the
    vertex is a branching point = 3 and more cells share it. Here we return
    a list (index) of such points and the cells that meet there.
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
    Produces terminals_map and a branch_map. Branch map has for each 
    branch(index) a collection of cells(indices) that make up the branch.
    Terminals map maps a branch index to a tuple of vertex indicies which
    are the branch boundaries.
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


def mesh_from_vc_data(vertices, cells, tol=0):
    '''
    Serial mesh from vertices: index -> coordinate,
                     cells: index -> tuple of vertex indices
    '''
    # Make sure that the vertices are unique
    assert tol == 0 or all(np.linalg.norm(xi - xj) > tol
                           for i, xi in enumerate(vertices) for xj in vertices[i+1:])
    # Sanity of cells
    vertices_per_cell = set(map(len, cells))
    assert len(vertices_per_cell) == 1
    vertices_per_cell = vertices_per_cell.pop()

    # FIXME: for now only line meshes
    tdim = {2: 1}[vertices_per_cell]
    cell_type = {(2, 1): 'interval'}[(vertices_per_cell, tdim)]
    
    nvertices, gdim = vertices.shape

    # Let's build the mesh
    mesh = Mesh()
    editor = MeshEditor()
    editor.open(mesh, cell_type, tdim, gdim)
    editor.init_vertices(nvertices)
    editor.init_cells(len(cells))

    for i, v in enumerate(vertices): editor.add_vertex(i, v)

    for i, c in enumerate(cells): editor.add_cell(i, *c)

    editor.close()

    # Check whether order was respected
    x = mesh.coordinates()
    assert tol == 0 or np.linalg.norm(x - vertices) < tol

    return mesh


def fit_circle(pts, method='enclose', ax=None):
    '''Find ellpse encloses/fits the points (Least-Squares CircleRandy Bullock)'''

    npts, dim = pts.shape
    assert dim == 2

    assert method in ('fit', 'enclose')

    x, y = pts.T
    xmean, ymean = np.mean(x), np.mean(y)

    if method == 'enclose':
        center = np.array([xmean, ymean])
        radius = np.sqrt(np.max(np.sum((pts - center)**2, 1)))

    else:
        u = x - xmean
        v = y - ymean

        cu, cv = np.linalg.solve(np.array([[np.sum(u*u), np.sum(u*v)],
                                           [np.sum(v*u), np.sum(v*v)]]),
                                 np.array([0.5*(np.sum(u*u*u) + np.sum(u*v*v)),
                                           0.5*(np.sum(v*v*v) + np.sum(v*u*u))]))

        center = np.array([cu + xmean, cv + ymean])

        radius = np.sqrt(cu**2 + cv**2 + (np.sum(u*u) + np.sum(v*v))/npts)

    if ax is not None:
        theta = np.linspace(0, 2*np.pi, 200)
        pts = np.c_[center[0] + radius*np.cos(theta),
                    center[1] + radius*np.sin(theta)]
        ax.plot(x, y, marker='x', linestyle='none')
        ax.plot(pts[:, 0], pts[:, 1], '-b')
        ax.plot(center[0], center[1], 'rx')

    return center, radius

# --------------------------------------------------------------------

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    npts = 400
    x0, y0, r = 1, 2, 3.5
    theta = np.linspace(0, 2*np.pi, npts)
    
    pts0 = np.c_[x0 + r*np.cos(theta), y0 + r*np.sin(theta)]
    noise = np.random.rand(npts, 2)*r/20.

    pts0 += noise
    
    x0, y0, r = fit_circle(pts0)
    pts = np.c_[x0 + r*np.cos(theta), y0 + r*np.sin(theta)]
    

    plt.figure()
    plt.plot(pts0[:, 0], pts0[:, 1], marker='x', linestyle='none')
    plt.plot(pts[:, 0], pts[:, 1], '-b')
    plt.axis('equal')
    plt.show()
