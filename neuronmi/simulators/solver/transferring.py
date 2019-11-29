import dolfin as df
import numpy as np


class SubMeshTransfer(object):
    '''Given a function space V over mesh and a different function space W 
    (not necessary of the same FEM family & degree) on the submesh this 
    object computes mapping from assigning from f in V to g in W and 
    the other way arround. This is done using the spatial correspondence
    of the dof coordinates of the spaces. Note that this might not be a 
    meaningful operation in some cases.
    '''
    def __init__(self, mesh, submesh):
        # The submesh must have come from EmbeddedMesh/SubMesh
        assert any((mesh.topology().dim() >= submesh.topology().dim(), 
                    mesh.num_cells() > submesh.num_cells()))

        assert df.MPI.size(mesh.mpi_comm()) == 1

        self.mesh = mesh
        self.submesh = submesh

        try:
            self.cell_map = submesh.parent_entity_map[mesh.id()][submesh.topology().dim()]
        except AttributeError:
            self.cell_map = submesh.data().array('parent_cell_indices',
                                                 submesh.topology().dim())
        self.cache = []

    def compute_map(self, toSpace, fromSpace, strict=True):
        '''
        Let f in toSpace and g in fromSpace. Return foo such that
        foo(f, g) asigns values of g to f.
        '''
        # Sanity of spaces
        assert not strict or toSpace.ufl_element().family() == fromSpace.ufl_element().family()
        assert not strict or toSpace.ufl_element().degree() == fromSpace.ufl_element().degree()

        assert toSpace.mesh().id() in (self.mesh.id(), self.submesh.id())
        assert fromSpace.mesh().id() in (self.mesh.id(), self.submesh.id())

        # We compute (and hold in cache the dof index map from assigning
        # from submesh to mesh
        if fromSpace.mesh().id() == self.submesh.id():
            if not len(self.cache):
                self.cache = self.__compute_map(submSpace=fromSpace, mSpace=toSpace)
            mapping = self.cache

            def foo(f, g):
                assert f.function_space() == toSpace
                assert g.function_space() == fromSpace

                f_values = f.vector().get_local()
                f_values[mapping] = g.vector().get_local()
                f.vector().set_local(f_values)
                f.vector().apply('insert')
                return f
        else:
            if not len(self.cache):
                self.cache = self.__compute_map(submSpace=toSpace, mSpace=fromSpace)
            mapping = self.cache
            
            def foo(f, g):
                assert f.function_space() == toSpace
                assert g.function_space() == fromSpace

                f_values = f.vector().get_local()
                f_values[:] = g.vector().get_local()[mapping]
                f.vector().set_local(f_values)
                f.vector().apply('insert')
                return f
        return foo

    def __compute_map(self, submSpace, mSpace):
        '''Comute the dof index map'''
        sub_dm = submSpace.dofmap()
        m_dm = mSpace.dofmap()

        sub_mesh = submSpace.mesh()
        m_mesh = mSpace.mesh()
        # subSpace cell -> mSpace entity -> mSpace cell
        mSpace.mesh().init(sub_mesh.topology().dim(), m_mesh.topology().dim())
        me2c = m_mesh.topology()(sub_mesh.topology().dim(), m_mesh.topology().dim())

        sub_dofs_x = submSpace.tabulate_dof_coordinates().reshape((submSpace.dim(), -1))
        m_dofs_x = mSpace.tabulate_dof_coordinates().reshape((mSpace.dim(), -1))

        # We shall compare dofs in some relative scale which is determined
        # by the size of mesh
        h_scale = max(sub_mesh.coordinates().max(axis=0)-sub_mesh.coordinates().min(axis=0))

        # subdofs to dofs
        mapping = np.nan*np.ones(submSpace.dim())
        for cell in df.cells(submSpace.mesh()):
            ci = cell.index()
            sub_dofs = list(sub_dm.cell_dofs(ci))
            # Pick up the entity of the of the mmesh where the subcell
            # came from
            entity = self.cell_map[ci]
            # Now we pick up all to dofs in the mspace which are assoc
            # with the cells connected to teh entitiy
            m_dofs = list(set(sum((list(m_dm.cell_dofs(mcell)) for mcell in me2c(entity)),
                                  [])))
            # Now look for a corresponding dof, where the correspondence
            # is defined by dofs sitting in the same spatial point
            # print m_dofs_x[m_dofs]
            while sub_dofs:
                s_dof = sub_dofs.pop()
                # Assigned only once
                if not np.isnan(mapping[s_dof]): continue
                # s dof location
                x = sub_dofs_x[s_dof]
                # print '\t', x
                # Closest mdof
                m_dof = min(m_dofs, key=lambda dof: np.linalg.norm(x-m_dofs_x[dof]))
                # There MUST be one which matches 'exactly'
                dof_x_error = np.linalg.norm(x-m_dofs_x[m_dof])/h_scale
                assert dof_x_error < 10*df.DOLFIN_EPS, dof_x_error
                # Insert
                mapping[s_dof] = m_dof
        # And we foudn them all 
        assert not list(filter(np.isnan, mapping))
                
        return np.array(mapping, dtype=int)
