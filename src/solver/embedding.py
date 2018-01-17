from collections import defaultdict
import dolfin as df
import numpy as np


def EmbeddedMesh(marking_function, markers):
    '''
    Construct a mesh of marked entities in marking_function.
    The output is the mesh and the cell function which inherited the markers. 
    The returned mesh has an antribute `entity_map` which is a map of new 
    mesh vertices to the old one, and new mesh cells to the old mesh entities.
    '''
    base_mesh = marking_function.mesh()
    # Prevent cell function (just not to duplicate functionality
    assert base_mesh.topology().dim() != marking_function.dim(), 'Use SubMesh'

    gdim = base_mesh.geometry().dim()
    tdim = marking_function.dim()
    assert tdim > 0, 'No Embedded mesh from vertices'

    if isinstance(markers, int): markers = [markers]

    base_mesh.init(tdim, 0)
    # Collect unique vertices based on their new-mesh indexing, the cells
    # of the embedded mesh are defined in terms of their embedded-numbering
    new_vertices, new_cells = [], []
    # NOTE: new_vertices is actually new -> old vertex map
    # Map from cells of embedded mesh to tdim entities of base mesh, and
    cell_map = []
    cell_colors = defaultdict(list)  # Preserve the markers

    new_cell_index, new_vertex_index = 0, 0
    for marker in markers:
        for entity in df.SubsetIterator(marking_function, marker):
            vs = entity.entities(0)
            cell = []
            # Vertex lookup
            for v in vs:
                try:
                    local = new_vertices.index(v)
                except ValueError:
                    local = new_vertex_index
                    new_vertices.append(v)
                    new_vertex_index += 1
                # Cell, one by one in terms of vertices
                cell.append(local)
            # The cell
            new_cells.append(cell)
            # Into map
            cell_map.append(entity.index())
            # Colors
            cell_colors[marker].append(new_cell_index)

            new_cell_index += 1

    # With acquired data build the mesh
    new_mesh = df.Mesh()
    editor = df.MeshEditor()

    if df.__version__ == '2017.2.0':
        cell_type = {1: 'interval', 2: 'triangle'}[tdim]
        editor.open(new_mesh, cell_type, tdim, gdim)            
    else:
        editor.open(new_mesh, tdim, gdim)
        
    editor.init_vertices(len(new_vertices))
    editor.init_cells(len(new_cells))

    vertex_coordinates = base_mesh.coordinates()[new_vertices]

    for vi, x in enumerate(vertex_coordinates): editor.add_vertex(vi, x)

    for ci, c in enumerate(new_cells): editor.add_cell(ci, *c)

    editor.close()

    # The entity mapping attriebute
    new_mesh.entity_map = {0: new_vertices, tdim: cell_map}

    f = df.MeshFunction('size_t', new_mesh, tdim, 0)
    f_ = f.array()
    # Finally the inherited marking function
    if len(markers) > 1:
        for marker, cells in cell_colors.iteritems(): f_[cells] = marker
    else:
        f.set_all(markers[0])
        
    return new_mesh, f

# -------------------------------------------------------------------

if __name__ == '__main__':
    mesh = df.UnitCubeMesh(10, 10, 10)

    f = df.MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
    chi = df.CompiledSubDomain('near(x[i], 0.5)', i=0) 
    for i in range(3):
        chi.i=i
        chi.mark(f, i+1)

    mesh, f = EmbeddedMesh(f, [1, 2, 3])

    df.File('f.pvd') << f
