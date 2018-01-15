import networkx as nx
import numpy as np
from itertools import izip

# FIXME: version which relies on gmsh For loops?
#        API

# This is a version which has the logic in python
def make_geo_branch(mesh, branch):
    '''
    Produce a geo file for branch of the mesh. Here a branch is a collection
    of cells + starting and ending vertices.
    '''
    assert hasattr(mesh, 'diam_info')
    
    # [cells], (vertex0, vertec1)
    branch, (start, stop) = branch

    c2v = mesh.topology()(1, 0)
    G = nx.Graph()
    G.add_edges_from((c2v(c) for c in branch))
    # Walk the branch
    path = [v for v in nx.algorithms.shortest_path(G, start, stop)]
    assert G.number_of_nodes() == len(path)

    # The branch is specified by vertices and raddi
    x = mesh.coordinates()[path]
    radii = mesh.diam_info[path]

    nsegments = len(x)-1

    segments = izip(izip(x[:-1], x[1:]), izip(radii[:-1], radii[1:]))

    # Load the definitions of segments
    with open('shape_lib_geo.txt') as f:
        code = ''.join(f.readlines())

        code = '\n'.join([code, 'index = 0;'])
        
    # Single cell will default to ...
    if nsegments == 1:
        code = '\n'.join([code, closed_closed(*next(segments))])
        # No need to add the union code
        return code

    # First
    code = '\n'.join([code, closed_open(*next(segments))])
    nsegments -= 1
    # Inner
    while nsegments > 1:
        code = '\n'.join([code, open_open(*next(segments))])
        nsegments -= 1
    # Last
    code = '\n'.join([code, open_closed(*next(segments))])
    nsegments -= 1
    assert nsegments == 0

    # Now the union code
    code = '\n'.join([code,
                      'nsegments = #segments[];',
                      'nsegments -= 1;',
                      'neuron = BooleanUnion {Volume{segments[0]}; Delete;}{Volume{segments[{1:nsegments}]}; Delete;};'])
    
    return code

# -----

def closed_closed((x0, x1), (r0, r1), tol=1E-12):
    return dispatch(abs(r0-r1) < tol, (0, 0), (x0, x1), (r0, r1))


def closed_open((x0, x1), (r0, r1), tol=1E-12):
    return dispatch(abs(r0-r1) < tol, (0, 1), (x0, x1), (r0, r1))


def open_closed((x0, x1), (r0, r1), tol=1E-12):
    return dispatch(abs(r0-r1) < tol, (1, 0), (x0, x1), (r0, r1))


def open_open((x0, x1), (r0, r1), tol=1E-12):
    return dispatch(abs(r0-r1) < tol, (1, 1), (x0, x1), (r0, r1))


def dispatch(is_cylinder, flow, (x0, x1), (r0, r1)):
    dx = x1 - x0
    parameters = {'base_x': x0[0], 'base_y': x0[1], 'base_z': x0[2],
                  'dir_x': dx[0], 'dir_y': dx[1], 'dir_z': dx[2]}
    
    if is_cylinder:
        parameters.update({'rad': r0})
        method = {(0, 0): 'ClosedCySegmentClosed',
                  (1, 0): 'CySegmentClosed',
                  (0, 1): 'ClosedCySegment',
                  (1, 1): 'CySegment'}[flow]
    else:
        parameters.update({'base_rad': r0, 'top_rad': r1})
        method = {(0, 0): 'ClosedSegmentClosed',
                  (1, 0): 'SegmentClosed',
                  (0, 1): 'ClosedSegment',
                  (1, 1): 'Segment'}[flow]

    code = '\n'.join(['%s = %g;' % kv for kv in parameters.items()])
    code = '\n'.join([code, 'Call %s;\n' % method])

    return code

# -------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import (File, CellFunction, SubMesh, Function, FunctionSpace,
                        dof_to_vertex_map)
    from dolfin_mesh import data_as_mesh, prune_mesh
    from common import find_branches
    
    path = 'L5_TTPC1_cADpyr232_1_geometry.npz'

    distance = 175
    nreduced = 0
    
    raw_mesh = data_as_mesh(path, round=8, distance=distance)

    # As a result of cliping the mesh at this point might have some
    # branches which are not connected to the rest.
    if distance < np.inf:
        graph = prune_mesh(raw_mesh)

        f = CellFunction('size_t', raw_mesh, 0)
        f.array()[graph] = 1

        # New we can build a mesh as a submesh
        mesh = SubMesh(raw_mesh, f, 1)
        mesh.type_info = raw_mesh.type_info[mesh.data().array('parent_cell_indices', 1)]
        mesh.diam_info = raw_mesh.diam_info[mesh.data().array('parent_vertex_indices', 0)]
        
        raw_mesh = mesh

    terminals, branches = find_branches(raw_mesh)
    print max(map(len, branches))

    # 3 here is just an example
    # branch = 3
    # code = make_geo_branch(mesh, (branches[branch], terminals[branch]))

    # print code
