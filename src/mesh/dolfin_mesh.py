from common import VertexCounter, mesh_from_vc_data, find_branches
import networkx as nx
import numpy as np


def data_as_mesh(path, round=0, distance=np.Inf):
    '''
    Represent the neuron segmentation as a line mesh in 3d where each
    cell corresponds to the slice (line orientation == cone(slice) orientation).
    The mesh also has maps which identify each cell as enum{soma, axon, dend} 
    and for each vertex we have the radius of the cone at that point.

    If distance < np.Inf only the part less then distance away from the 
    soma center is kept.
    '''
    # Sane npz file
    data = np.load(path)
    keys = ('bottoms', 'tops', 'diams', 'secs')
    assert all(key in data for key in keys)
    
    bottoms, tops, diams, secs = [data[k] for k in keys]
    # By rounding we bring close vertices to identical vertices
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

    # Prepare clipping
    if distance < np.Inf:
        # Compute approx center of the soma
        center = np.mean(tops[secs == 'soma'], axis=0)
        # Filter for points which are close
        is_inside = lambda x: np.linalg.norm(x-center) < distance
    else:
        is_inside = lambda x: True

    # Label vertices and define cell in terms of vertex ordering
    for (bottom, top, sec, diam_b, diam_t) in zip(bottoms[:n-1], tops[:n-1], secs[:-1], diams[:n-1], diams[1:n]):
        # Skip those outside
        if not(is_inside(bottom) and is_inside(top)): continue
        
        index_b = vc.insert(bottom)
        index_t = vc.insert(top)
        
        cells.append((index_b, index_t))
        cell_data.append(lookup[sec])

        # Duplicate (but should be robust)
        vertex_data[index_b] = diam_b
        vertex_data[index_t] = diam_t
    # Remember to take away the initialization guy
    vertices = vc.vertices[1:]

    # Now we can define the 1d_mesh
    raw_mesh = mesh_from_vc_data(vertices, cells)

    # Now add the info on cells, vertices
    f_values = np.array(cell_data, dtype='uintp')
    raw_mesh.type_info = f_values

    f_values = np.fromiter((vertex_data[index] for index in sorted(vertex_data.keys())), dtype=float)
    raw_mesh.diam_info = f_values

    return raw_mesh


def prune_mesh(mesh, tol=1E-12):
    '''
    A mesh with the isolated components removed. Isolated in the sense 
    that the neuron is supposed to be one electric circuit but clipping 
    the mesh might violate this assumption.
    '''
    terminals, branches = find_branches(mesh)
    # Figure out which branch is the soma
    soma_branch = set(np.where(mesh.type_info == 1)[0])
    soma_branch = [i for i, branch in enumerate(branches) if branch == soma_branch]
    assert len(soma_branch) == 1
    soma_branch = soma_branch.pop()
    
    # Fit ellipsoid to soma to figure out which branches would be connected
    # to it. NOTE: This assmes that soma data is @ z = 0
    x = mesh.coordinates()
    c2v = mesh.topology()(1, 0)
    soma_points = []
    for cell in branches[soma_branch]:
        #  o----t----o
        #       | dt
        #     o-b-o
        # Also extended in z direction
        top_x, bottom_x = x[c2v(cell)]
        assert abs(top_x[-1] < tol) and abs(bottom_x[-1] < tol)
        dt = top_x - bottom_x
        # Perpendicular dir
        dt_perp = np.array([-dt[1], dt[0], 0.]); dt_perp /= np.linalg.norm(dt_perp)
        plane_normal = np.array([0, 0, 1.])
        # Add points o
        top_r, bottom_r = mesh.diam_info[c2v(cell)]
        soma_points.extend([top_x - 0.5*top_r*dt_perp,
                            top_x + 0.5*top_r*dt_perp,
                            top_x - 0.5*top_r*plane_normal,
                            top_x + 0.5*top_r*plane_normal,
                            bottom_x - 0.5*bottom_r*dt_perp,
                            bottom_x + 0.5*bottom_r*dt_perp,
                            bottom_x - 0.5*bottom_r*plane_normal,
                            bottom_x + 0.5*bottom_r*plane_normal])
        
    A, b, c = ellipse_fit(np.array(soma_points))
    # FIXME
    # Ellipse is x'*A*x + b'x + c = 0. With <= 0 these are points inside
    inside_ellipse = lambda x: np.dot(x, A.dot(x)) + np.dot(b, x) + c <= tol
    # FIXME
    # A sanity check, top_x and bottom_x should be inside the ellipse

    insiders = set()
    for branch in (set(range(len(branches))) - set([soma_branch])):
        is_inside = [inside_ellipse(xi) for xi in x[terminals[branch]]]
        if any(is_inside):
            assert not all(is_inside)
            # Collect the inside points
            v0, v1 = terminals[branch]
            insiders.add(v0 if is_inside[0] else v1)
    insiders.update(terminals[soma_branch])
    insiders = list(insiders)

    terminals_ext = list(terminals)
    # In reality these branches are connected so we introduce there connections
    # to the graph as well
    for i, v0 in enumerate(insiders):
        for v1 in insides[i+1:]:
            terminals_ext.append((v0, v1))
    
    # What we are after now is the largest connected component of the graph
    G = nx.Graph()
    G.add_edges_from(terminals_ext)
    G = max(nx.connected_component_subgraphs(G), key=len)
    
    # FIXME: build the mesh

    
# --------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import File, CellFunction, SubsetIterator
    from common import find_branches, branch_terminals
    from itertools import imap
    import operator as op
    
    path = 'L5_TTPC1_cADpyr232_1_geometry.npz'

    distance = 50
    raw_mesh = data_as_mesh(path, round=8, distance=distance)
    # As a result of cliping the mesh at this point might have some
    # branches which are not connected to the rest.
    if distance < np.inf:
        graph = prune_mesh(raw_mesh)
    
    # Look for branches in the mesh that are isolated
    # These should be removed
    # if center_distance is not None:
    #     continue
    #     terminals, branches = find_branches(mesh)
    #     # Figure out which branch is the soma
    #     soma_branch = set(cell.index() for cell in SubsetIterator(mesh.type_info, 1))
        
    #     remove_branch = set()
    #     for i, (v0, v1) in enumerate(terminals):
    #         others = map(list, terminals[:i]) + map(list, terminals[i+1:])
    #         others = set(sum(others, []))
    #         # Will remove soma branch
    #         if v0 not in others and v1 not in others:
    #             remove_branch.add(i)
    #     # One of them is some
    #     soma_branch = set([i for i in remove_branch if branches[i] == soma_branch])
    #     assert soma_branch
    #     remove_branch = remove_branch - soma_branch
    #     f = CellFunction('size_t', mesh, 0)
        
    #     for branch in remove_branch:
    #         for cell in branches[branch]: f[int(cell)] = 1

    #     File('f.pvd') << f
            
    #     new_vertices, new_vertex_data = [], []
        
    #     new_cells, new_cell_values = [], []
            
    #     c2v = mesh.topology()(1, 0)
    #     for cell in imap(op.methodcaller('index'), SubsetIterator(f, 0)):
    #         v0, v1 = c2v(cell)
    #         print (v0, v1)
    #         print v0, v0 in new_vertices
    #         try:
    #             v0 = new_vertices.index(v0)
    #         except ValueError:
    #             new_vertices.append(v0)
    #             v0 = len(new_vertices) - 1
    #         print v0

    #         print v1, v1 in new_vertices
    #         try:
    #             v1 = new_vertices.index(v1)
    #         except ValueError:
    #             new_vertices.append(v1)
    #             v1 = len(new_vertices) - 1
    #         print v1
            
    #         new_cells.append((v0, v1))
    #         new_cell_values.append(mesh.type_info[cell])
            
    #     # FIXME:

    #     x = mesh.coordinates()
    #     # Mesh wthout islands
    #     tdim = 1  # intervals
    #     gdim = 3  # embedded in R^3
    #     # Let's build the mesh
    #     meshr = Mesh()
    #     editor = MeshEditor()
    #     editor.open(meshr, 'interval', tdim, gdim)
    #     editor.init_vertices(len(new_vertices))
    #     editor.init_cells(len(new_cells))

    #     for i, v in enumerate(new_vertices):
    #         xi = x[v]
    #         editor.add_vertex(i, xi)
    #         new_vertex_data.append(mesh.diam_info(xi))

    #     for i, c in enumerate(new_cells): editor.add_cell(i, *c)

    #     editor.close()

    #     # Preserve type info
    #     f_values = np.array(new_cell_values, dtype='uintp')
    #     f = MeshFunction('size_t', meshr, tdim, 0)
    #     f.set_values(f_values)
    #     meshr.type_info = f

    #     # Preserve vertex info
    #     new_vertex_data = np.array(new_vertex_data, dtype=float)
    #     V = FunctionSpace(meshr, 'CG', 1)
    #     f = Function(V)
    #     f.vector().set_local(new_vertex_data[dof_to_vertex_map(V)])
    #     meshr.diam_info = f

    #     # how/doc
    #     # ---------
    #     # Simple case
    #     # Ellipse fit
    #     # The real deal
        
    #     File('bar.pvd') << meshr.diam_info
    # # We have the geometry + data representing the radii associated with
    # # vertices + data representing the type associated with each cell
    # File('test.pvd') << mesh.diam_info
