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
            
# --------------------------------------------------------------------

if __name__ == '__main__':
    from embedding import EmbeddedMesh
    import sys

    try:
        n = int(sys.argv[1])
    except (ValueError, IndexError):
        n = 8
    
    mesh = df.UnitCubeMesh(*(n, )*3)
    # Submesh defined by three markers
    marking_f = df.MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
    chi = df.CompiledSubDomain('near(x[i], 0.5)', i=0) 
    for i in range(3):
        chi.set_property('i', i)
        chi.mark(marking_f, i+1)

    markers = [1, 2, 3]
    submesh = EmbeddedMesh(marking_f, markers)
    f = submesh.marking_function
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
    print(df.sqrt(df.assemble(error_form)))


    # ----------------------------------------------------------------

    mesh = df.UnitSquareMesh(n, n)

    f = df.MeshFunction('size_t', mesh, 1, 0)
    df.CompiledSubDomain('near(x[0], 0)').mark(f, 1)
    df.CompiledSubDomain('near(x[0], 1)').mark(f, 2)
    df.CompiledSubDomain('near(x[1], 0)').mark(f, 3)
    df.CompiledSubDomain('near(x[1], 1)').mark(f, 4)

    parent = EmbeddedMesh(f, [1, 2, 3, 4])
    subdomains = parent.marking_function

    child0 = EmbeddedMesh(subdomains, [1, 2])
    child1 = EmbeddedMesh(subdomains, [3, 4])

    V, V0, V1 = (df.FunctionSpace(m, 'DG', 0) for m in (parent, child0, child1))
    
    transfer0 = SubMeshTransfer(parent, child0)
    transfer1 = SubMeshTransfer(parent, child1)
    
    toV0_fromV = transfer0.compute_map(V0, V)
    toV_fromV0 = transfer0.compute_map(V, V0)

    toV1_fromV = transfer1.compute_map(V1, V)
    toV_fromV1 = transfer1.compute_map(V, V1)

    expr = df.Expression('x[0] + x[1]', degree=1)
      
    f0 = df.interpolate(expr, V0)
    f1 = df.interpolate(expr, V1)

    f = df.Function(V)
    x, y = df.SpatialCoordinate(parent)
    
    f = toV_fromV0(f, f0)
    print(df.assemble(df.inner(f - (x+y), f-(x+y))*df.dx))
    
    f = toV_fromV1(f, f1)
    print(df.assemble(df.inner(f - (x+y), f-(x+y))*df.dx))

    # Go the other way
    f0.vector().zero()
    f1.vector().zero()

    f0 = toV0_fromV(f0, f)
    x0, y0 = df.SpatialCoordinate(V0.mesh())
    print(df.assemble(df.inner(f0 - (x0+y0), f0-(x0+y0))*df.dx))

    f1 = toV1_fromV(f1, f)
    x1, y1 = df.SpatialCoordinate(V1.mesh())
    print(df.assemble(df.inner(f1 - (x1+y1), f1-(x1+y1))*df.dx))

    W = df.FunctionSpace(mesh, 'Discontinuous Lagrange Trace', 0)
    transfer = SubMeshTransfer(mesh, parent)    
    toW_fromV = transfer.compute_map(W, V, strict=False)

    w = df.Function(W)
    toW_fromV(w, f)

    x, y = df.SpatialCoordinate(W.mesh())
    print(df.assemble(df.inner(w - (x+y), w -(x+y))*df.ds))
    

