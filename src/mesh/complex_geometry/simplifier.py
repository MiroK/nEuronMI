from common import find_branches, mesh_from_vc_data
from collections import defaultdict
import networkx as nx
import numpy as np


def graph_analysis(terminals):
    '''Which branches are loops? Which vertices are branching/end points?'''
    loops = set(branch for branch, (v0, v1) in enumerate(terminals) if v0 == v1)
    not_loops = set(range(len(terminals))) - loops

    # vertex -> branches
    v2b = defaultdict(list)
    [v2b[v].append(branch) for branch in not_loops for v in terminals[branch]]
    end_points = [v for (v, b) in v2b.iteritems() if len(b) == 1]

    # As a map which knows about the connected branches
    branch_points = {v: b for v, b in v2b.iteritems() if v not in end_points}

    return {'loops': loops, 'end_points': end_points, 'branch_points': branch_points}


def simplify_branch(mesh, branch):
    '''Skip every other vertex'''
    # Only line meshes with consistent data
    assert mesh.topology().dim() == 1

    branch, (start, stop) = branch

    # A single cell note cannot by simplified any further
    if len(branch) == 1:
        return [start, stop], mesh.type_info[int(next(iter(branch)))]

    branch = map(int, branch)
    branch_type = set(mesh.type_info[branch])
    assert len(branch_type) == 1
    branch_type = branch_type.pop()
    
    c2v = mesh.topology()(1, 0)
    G = nx.Graph()
    G.add_edges_from((c2v(c) for c in branch))
    # Walk the branch
    path = [v for v in nx.algorithms.shortest_path(G, start, stop)]
    assert G.number_of_nodes() == len(path)
    
    # Skip
    path = path[::2] + ([stop] if len(path) % 2 == 0 else [])

    assert path[0] == start and path[-1] == stop, ((path[0], start), (path[-1], stop), path, G.edges())

    return path, branch_type

# FIXME: collisions? - just detect and bailout
#        be aware of identities, i.e. detect states that cannot be simplified
def simplify_mesh(mesh, ntimes=0):
    '''Straighten branches by skiping points'''
    assert ntimes >= 0

    if ntimes == 0: return mesh

    # Do one simplification
    terminals, branches = find_branches(mesh)
    # Sanity check for correct branching is not loops
    analysis_data = graph_analysis(terminals)
    assert not analysis_data['loops']

    end_points, branch_points = analysis_data['end_points'], analysis_data['branch_points']

    vertex_indices = []  # index(new mesh) -> old-mesh-index
    next_index = 0
    
    cells = []           # of new mesh in new mesh index ordering
    cell_data = []
    # Gather data for simplified mesh
    for i, branch in enumerate(branches):
        G, btype = simplify_branch(mesh, (branch, terminals[i]))
        # Figure out vertex indexing
        Gstart = G[0]
        Ginterior = G[1:-1]
        Gstop = G[-1]

        # Local
        branch_vertex_indices = []

        # I do this in order of points (might be better for sparsity)
        if Gstart in end_points: # Guaranteed to be unique
            vertex_indices.append(Gstart)
            
            branch_vertex_indices.append(next_index)
            next_index += 1
        else:
            # Need to look it up
            assert Gstart in branch_points
            try:
                branch_vertex_indices.append(vertex_indices.index(Gstart))
            except ValueError:
                vertex_indices.append(Gstart)
                
                branch_vertex_indices.append(next_index)
                next_index += 1

        # All the interior points are unique
        vertex_indices.extend(Ginterior)
        # Local
        branch_vertex_indices.extend(range(next_index, next_index + len(Ginterior)))
        next_index += len(Ginterior)

        if Gstop in end_points: # Guaranteed to be unique
            vertex_indices.append(Gstop)
            
            branch_vertex_indices.append(next_index)
            next_index += 1
        else:
            # Need to look it up
            assert Gstop in branch_points
            try:
                branch_vertex_indices.append(vertex_indices.index(Gstop))
            except ValueError:
                vertex_indices.append(Gstop)

                branch_vertex_indices.append(next_index)
                next_index += 1
        # With the computed vertex numbering we can make cells
        cells.extend([(v0, v1) for v0, v1 in zip(branch_vertex_indices[:-1],
                                                 branch_vertex_indices[1:])])
        cell_data.extend([btype]*(len(G)-1))
    # New
    simplified_mesh = mesh_from_vc_data(mesh.coordinates()[vertex_indices], cells)
    simplified_mesh.diam_info = mesh.diam_info[vertex_indices]
    simplified_mesh.type_info = np.array(cell_data, dtype='uintp')
    
    # Recurse
    return simplify_mesh(simplified_mesh, ntimes-1)


# -------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import (File, CellFunction, SubMesh, Function, FunctionSpace,
                        dof_to_vertex_map)
    from dolfin_mesh import data_as_mesh, prune_mesh
    
    path = 'L5_TTPC1_cADpyr232_1_geometry.npz'

    distance = 175
    nreduced = 0
    
    raw_mesh = data_as_mesh(path, round=8, distance=distance)

    # As a result of cliping the mesh at this point might have some
    # branches which are not connected to the rest.
    if distance < np.inf:
        graph = prune_mesh(raw_mesh)
        # Viz it
        f = CellFunction('size_t', raw_mesh, 0)
        f.array()[graph] = 1

        File('results/cc_graph_reduced%d_d%g.pvd' % (nreduced, distance)) << f
        # New we can build a mesh as a submesh
        mesh = SubMesh(raw_mesh, f, 1)
        mesh.type_info = raw_mesh.type_info[mesh.data().array('parent_cell_indices', 1)]
        mesh.diam_info = raw_mesh.diam_info[mesh.data().array('parent_vertex_indices', 0)]
        
        raw_mesh = mesh

    terminals, branches = find_branches(raw_mesh)
        
    analysis_data = graph_analysis(terminals)

    f = CellFunction('size_t', mesh, 0); f.set_values(mesh.type_info)
    File('before.pvd') << f

    mesh = simplify_mesh(mesh, ntimes=4)

    f = CellFunction('size_t', mesh, 0); f.set_values(mesh.type_info)
    File('after.pvd') << f

