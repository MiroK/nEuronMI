from common import (VertexCounter, mesh_from_vc_data, find_branches, fit_circle)
import operator as op
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


def reduce_mesh(mesh, tol=1E-12):
    '''
    Produce cells that represent mesh with the isolated components removed. 
    Isolated in the sense that the neuron is supposed to be one electric 
    circuit but clipping the mesh might violate this assumption.
    '''
    terminals, branches = find_branches(mesh)
    # Figure out which branch is the soma
    soma_branch = set(np.where(mesh.type_info == 1)[0])
    soma_branch = [i for i, branch in enumerate(branches) if branch == soma_branch]
    assert len(soma_branch) == 1
    soma_branch = soma_branch.pop()
    
    # Fit circle to soma to figure out which branches would be connected
    # to it. NOTE: This assmes that soma data is @ z = 0
    x = mesh.coordinates()
    c2v = mesh.topology()(1, 0)
    soma_points = []
    for cell in branches[soma_branch]:
        #  o----t----o
        #       | dt
        #     o-b-o
        top_x, bottom_x = x[c2v(cell)]
        assert abs(top_x[-1] < tol) and abs(bottom_x[-1] < tol)
        dt = top_x - bottom_x
        # Perpendicular dir
        dt_perp = np.array([-dt[1], dt[0], 0.]); dt_perp /= np.linalg.norm(dt_perp)
        # Add points o
        top_r, bottom_r = mesh.diam_info[c2v(cell)]
        soma_points.extend([top_x - 0.5*top_r*dt_perp,
                            top_x + 0.5*top_r*dt_perp,
                            bottom_x - 0.5*bottom_r*dt_perp,
                            bottom_x + 0.5*bottom_r*dt_perp])

    (center_x, center_y), radius = fit_circle(np.array(soma_points)[:, :2])

    # These are the branches
    # Center_z is - 
    is_inside_sphere = lambda x: (x[0]-center_x)**2 + (x[1]-center_y)**2 + x[2]**2 - radius**2 < tol
    # A sanity check, top_x and bottom_x should be inside the ellipse    
    assert all(is_inside_sphere(xi) for cell in branches[soma_branch] for xi in x[c2v(cell)])

    insiders = set()
    for branch in (set(range(len(branches))) - set([soma_branch])):
        is_inside = [is_inside_sphere(x[vi]) for vi in terminals[branch]]
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
        for v1 in insiders[i+1:]:
            terminals_ext.append((v0, v1))
    
    # What we are after now is the largest connected component of the graph
    G = nx.Graph()
    G.add_edges_from(terminals_ext)
    G = max(nx.connected_component_subgraphs(G), key=len)
    # These are candidates (because of the extra artif soma connects)
    circuit_branches = map(lambda x: tuple(sorted(x)), G.edges())

    # Collect all the cells of the ciruit
    circuit_cells = reduce(op.or_, (branches[branch]
                                    for branch, ts in enumerate(terminals)
                                    if tuple(sorted(ts)) in circuit_branches))

    return list(circuit_cells)
    
# --------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import (File, CellFunction, SubMesh, Function, FunctionSpace,
                        dof_to_vertex_map)
    
    path = 'L5_TTPC1_cADpyr232_1_geometry.npz'

    distance = 175
    raw_mesh = data_as_mesh(path, round=8, distance=distance)

    # As a result of cliping the mesh at this point might have some
    # branches which are not connected to the rest.
    if distance < np.inf:
        graph = reduce_mesh(raw_mesh)
        # Viz it
        f = CellFunction('size_t', raw_mesh, 0)
        f.array()[graph] = 1

        File('cc_graph.pvd') << f
        # New we can build a mesh as a submesh
        mesh = SubMesh(raw_mesh, f, 1)
        mesh.type_info = raw_mesh.type_info[mesh.data().array('parent_cell_indices', 1)]
        mesh.diam_info = raw_mesh.diam_info[mesh.data().array('parent_vertex_indices', 0)]
        
        raw_mesh = mesh
    
    # Represent *_info as functions. To be used for mesh generation
    segment_types = CellFunction('size_t', mesh, 0)
    segment_types.set_values(mesh.type_info)
    # P1 for plotting
    V = FunctionSpace(mesh, 'CG', 1)
    diam_info = Function(V)
    diam_info.vector().set_local(mesh.diam_info[dof_to_vertex_map(V)])

    File('segment_types.pvd') << segment_types
    File('diam_info.pvd') << diam_info

    # FIXME: build gmsh file which represents the neuron
