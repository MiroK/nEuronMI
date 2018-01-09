from collections import defaultdict
from common import find_branches
import networkx as nx
import numpy as np


def graph_analysis(terminals):
    '''Which branches are loops? Which vertices are branching/end points?'''
    loops = set(branch for branch, (v0, v1) in enumerate(terminals) if v0 == v1)
    not_loops = set(range(len(terminals))) - loops

    # vertex -> branches
    v2b = defaultdict(list)
    [v2b[v].append(branch) for branch in not_loops for v in terminals[branch]]
    print terminals
    print v2b
    end_points = [v for (v, b) in v2b.iteritems() if len(b) == 1]

    # As a map which knows about the connected branches
    branch_points = {v: b for v, b in v2b.iteritems() if v not in end_points}

    return {'loops': loops, 'end_points': end_points, 'branch_points': branch_points}


def simplify_branch(mesh, branch):
    '''Skip every other vertex'''
    # Only line meshes with consistent data
    assert mesh.topology().dim() == 1

    branch, (start, stop) = branch
    
    branch = map(int, branch)
    branch_type = set(mesh.type_info[branch])
    assert len(branch_type) == 1
    branch_type = branch_type.pop()
    
    c2v = mesh.topology()(1, 0)
    G = nx.Graph()
    G.add_edges_from((c2v(c) for c in branch))
    # Walk the branch
    path = [v for v in nx.algorithms.shortest_path(G)]
    assert G.number_of_nodes() == len(path)
    # Skip
    path = path[::2]
    if len(path) % 2 == 0: path.append(stop)

    assert path[0] == start and path[-1] == stop

    return path, branch_type


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

    for i, branch in enumerate(branches):
        G = simplify_branch(mesh, (branches, terminals[i]))

    # FIXME: finish this/demo
    # FIXME: geo for branch


    # Recurse
    return simplify_mesh(mesh, ntimes-1)


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

    G = simplify_branch(mesh, (branches[1], terminals[1]))
