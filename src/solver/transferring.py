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
        # FIXME: own class?
        assert hasattr(submesh, 'entity_map') or isinstance(submesh, df.SubMesh)
        assert any((mesh.topology().dim() >= submesh.topology().dim(), 
                    mesh.num_cells() > submesh.num_cells()))

        # FIXME: serial only (for now)
        assert df.MPI.size(mesh.mpi_comm()) == 1

        self.mesh = mesh
        self.submesh = submesh

        try:
            self.cell_map = submesh.entity_map[submesh.topology().dim()]
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
        # In general we assume that the submspace is smaller
        assert submSpace.dim() < mSpace.dim()

        sub_dm = submSpace.dofmap()
        m_dm = mSpace.dofmap()

        sub_mesh = submSpace.mesh()
        m_mesh = mSpace.mesh()
        # subSpace cell -> mSpace entity -> mSpace cell
        mSpace.mesh().init(sub_mesh.topology().dim(), m_mesh.topology().dim())
        me2c = m_mesh.topology()(sub_mesh.topology().dim(), m_mesh.topology().dim())

        sub_dofs_x = submSpace.tabulate_dof_coordinates().reshape((submSpace.dim(), -1))
        m_dofs_x = mSpace.tabulate_dof_coordinates().reshape((mSpace.dim(), -1))

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
                # There MUST be one which matches exactly
                assert np.linalg.norm(x-m_dofs_x[m_dof]) < 10*df.DOLFIN_EPS, np.linalg.norm(x-m_dofs_x[m_dof])
                # Insert
                mapping[s_dof] = m_dof
        # And we foudn them all 
        assert not filter(np.isnan, mapping)
                
        return np.array(mapping, dtype=int)
            
# --------------------------------------------------------------------

if __name__ == '__main__':
    from embedding import EmbeddedMesh
    import sys

    try:
        n = int(sys.argv[1])
    except ValueError:
        n = 8
    
    mesh = df.UnitCubeMesh(*(n, )*3)
    # Submesh defined by three markers
    marking_f = df.MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
    chi = df.CompiledSubDomain('near(x[i], 0.5)', i=0) 
    for i in range(3):
        chi.i=i
        chi.mark(marking_f, i+1)

    markers = [1, 2, 3]
    submesh, f = EmbeddedMesh(marking_f, markers)
    # Two function space of the respected meshes
    V = df.FunctionSpace(mesh, 'CG', 1)
    W = df.FunctionSpace(submesh, 'CG', 1)

    transferer = SubMeshTransfer(mesh, submesh)
    # to W from V
    assign_toW_fromV = transferer.compute_map(W, V, strict=False)

    # The truth
    f = df.Expression('sin(pi*(x[0]+2*x[1]+3*x[2]))', degree=2)
    # Data at V
    fV = df.interpolate(f, V)
    fW = df.Function(W)
    # assigned to W
    assign_toW_fromV(fW, fV)
    # The end results should be like fWO
    fW0 = df.interpolate(f, W)
    # And?
    assert (fW.vector() - fW0.vector()).norm('linf') < 1E-10*W.dim()

    # Now, the other way around. Perturb the W
    fW.vector()[:] *= -1

    assign_toV_fromW = transferer.compute_map(V, W)
    # Put W data to V
    fV = df.Function(V)
    assign_toV_fromW(fV, fW)
    # Here the check makes sure that only the points on the manifold are used
    error = max(abs(fV(x) - fW(x)) for x in submesh.coordinates())
    assert error < 1E-10*V.dim()

    # ----------------------------------------------------------------
    # In main application we want a map between DLT and DG
    V = df.FunctionSpace(mesh, 'Discontinuous Lagrange Trace', 0)
    W = df.FunctionSpace(submesh, 'DG', 0)

    transferer = SubMeshTransfer(mesh, submesh)
    # to W from V
    assign_toW_fromV = transferer.compute_map(W, V, strict=False)

    # The truth
    f = df.Expression('sin(pi*(x[0]+2*x[1]+3*x[2]))', degree=2)
    # Data at V
    fV = df.interpolate(f, V)
    fW = df.Function(W)
    # assigned to W
    assign_toW_fromV(fW, fV)
    # The end results should be like fWO
    fW0 = df.interpolate(f, W)
    # And?
    assert (fW.vector() - fW0.vector()).norm('linf') < 1E-10*W.dim()
    
    # Now, the other way around. Perturb the W
    fW.vector()[:] *= -1

    assign_toV_fromW = transferer.compute_map(V, W, strict=False)
    # Put W data to V
    fV = df.Function(V)
    assign_toV_fromW(fV, fW)
    # The test is based on integration of the error
    f = -df.Expression('sin(pi*(x[0]+2*x[1]+3*x[2]))', degree=2)
    # In the absend of point evals for DLT space we only check here
    # for convergence of the error = This is the case; the error is halved
    # on mesh refinement
    dS_ = df.Measure('dS', subdomain_data=marking_f, domain=mesh)
    error_form = sum((fV('+')-f('+'))**2*dS_(i) for i in markers)
    print df.sqrt(df.assemble(error_form))
