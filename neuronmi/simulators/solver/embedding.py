from neuronmi.simulators.solver.make_mesh_cpp import make_mesh
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
        e2v = base_mesh.topology()(tdim, 0)
        # Collect unique vertices based on their new-mesh indexing, the cells
        # of the embedded mesh are defined in terms of their embedded-numbering
        new_vertices, new_cells = [], []
        # NOTE: new_vertices is actually new -> old vertex map
        # Map from cells of embedded mesh to tdim entities of base mesh, and
        cell_map = []
        cell_colors = defaultdict(list)  # Preserve the markers

        marking_function_arr = marking_function.array()
        
        new_cell_index, new_vertex_index = 0, 0
        for marker in markers:
            for entity in np.where(marking_function_arr == marker)[0]:
                vs = list(e2v(entity))
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
                cell_map.append(entity)
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
