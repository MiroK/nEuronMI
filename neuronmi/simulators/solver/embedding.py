from .make_mesh_cpp import make_mesh
from collections import defaultdict
from itertools import chain
import dolfin as df
import numpy as np
import operator


class EmbeddedMesh(df.Mesh):
    '''
    Construct a mesh of marked entities in marking_function.
    The output is the mesh with cell function which inherited the markers. 
    and an antribute `parent_entity_map` which is dict with a map of new 
    mesh vertices to the old ones, and new mesh cells to the old mesh entities.
    Having several maps in the dict is useful for mortaring.
    '''
    def __init__(self, marking_function, markers):
        if not isinstance(markers, (list, tuple)): markers = [markers]
        
        # Convenience option to specify only subdomains
        is_number = lambda m: isinstance(m, int)
        new_markers = []
        # Build a new list int list with facet_function marked
        if not all(map(is_number, markers)):
            
            numbers = filter(is_number, markers)
            next_int_marker = max(numbers) if numbers else 0
            for marker in markers:
                if is_number(marker):
                    new_markers.append(marker)
                else:
                    next_int_marker += 1
                    # SubDomain
                    try:
                        marker.mark(marking_function, next_int_marker)
                    except AttributeError:
                        # A string
                        try:
                            df.CompiledSubDomain(marker).mark(marking_function, next_int_marker)
                        # A lambda
                        except TypeError:
                            df.CompiledSubDomain(*marker()).mark(marking_function, next_int_marker)

                    new_markers.append(next_int_marker)
            
            markers = new_markers
            
        base_mesh = marking_function.mesh()

        assert base_mesh.topology().dim() >= marking_function.dim()

        # Work in serial only (much like submesh)
        assert df.MPI.size(base_mesh.mpi_comm()) == 1

        gdim = base_mesh.geometry().dim()
        tdim = marking_function.dim()
        assert tdim > 0, 'No Embedded mesh from vertices'

        assert markers, markers

        # We reuse a lot of Submesh capabilities if marking by cell_f
        if base_mesh.topology().dim() == marking_function.dim():
            # Submesh works only with one marker so we conform
            color_array = marking_function.array()
            color_cells = dict((m, np.where(color_array == m)[0]) for m in markers)

            # So everybody is marked as 1
            one_cell_f = df.MeshFunction('size_t', base_mesh, tdim, 0)
            for cells in color_cells.values(): one_cell_f.array()[cells] = 1
            
            # The Embedded mesh now steals a lot from submesh
            submesh = df.SubMesh(base_mesh, one_cell_f, 1)

            df.Mesh.__init__(self, submesh)

            # The entity mapping attribute;
            # NOTE: At this point there is not reason to use a dict as
            # a lookup table
            mesh_key = marking_function.mesh().id()
            mapping_0 = submesh.data().array('parent_vertex_indices', 0)

            mapping_tdim = submesh.data().array('parent_cell_indices', tdim)
            self.parent_entity_map = {mesh_key: {0: dict(enumerate(mapping_0)),
                                                 tdim: dict(enumerate(mapping_tdim))}}
            # Finally it remains to preserve the markers
            f = df.MeshFunction('size_t', self, tdim, 0)
            f_values = f.array()
            if len(markers) > 1:
                old2new = dict(zip(mapping_tdim, range(len(mapping_tdim))))
                for color, old_cells in color_cells.items():
                    new_cells = np.array([old2new[o] for o in old_cells], dtype='uintp')
                    f_values[new_cells] = color
            else:
                f.set_all(markers[0])
            
            self.marking_function = f
            # Declare which tagged cells are found
            self.tagged_cells = set(markers)
            # https://stackoverflow.com/questions/2491819/how-to-return-a-value-from-init-in-python            
            return None  

        # Otherwise the mesh needs to by build from scratch
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
        vertex_coordinates = base_mesh.coordinates()[new_vertices]
        new_cells = np.array(new_cells, dtype='uintp')
        
        # With acquired data build the mesh
        df.Mesh.__init__(self)
        # Fill
        make_mesh(coordinates=vertex_coordinates, cells=new_cells, tdim=tdim, gdim=gdim,
                  mesh=self)

        # The entity mapping attribute
        mesh_key = marking_function.mesh().id()
        self.parent_entity_map = {mesh_key: {0: dict(enumerate(new_vertices)),
                                             tdim: dict(enumerate(cell_map))}}

        f = df.MeshFunction('size_t', self, tdim, 0)
        f_ = f.array()
        # Finally the inherited marking function
        if len(markers) > 1:
            for marker, cells in cell_colors.items():
                f_[cells] = marker
        else:
            f.set_all(markers[0])

        self.marking_function = f
        # Declare which tagged cells are found
        self.tagged_cells = set(markers)

        
# --------------------------------------------------------------------


if __name__ == '__main__':
    import dolfin as df 
    mesh = df.UnitCubeMesh(10, 10, 10)

    f = df.MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
    chi = df.CompiledSubDomain('near(x[i], 0.5)', i=0) 

    for i in range(3):
        chi.set_property('i', i)
        chi.mark(f, i+1)

    mesh = EmbeddedMesh(f, [1, 2, 3])
    
    df.File('f.pvd') << mesh.marking_function
