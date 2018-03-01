from dolfin import *


def LinearSystemSolver(A, W, solver_parameters):
    '''
    This is a dispatches to solvers for EMI model. It returns something
    cable of solve(x, b) call.
    '''
    if solver_parameters['linear_solver'] == 'direct':
        return direct_solver(A, W, solver_parameters)
    # Experimental solver which solves the system w/out the constrained dofs
    elif solver_parameters['linear_solver'] == 'direct_reduced':
        cstr_dofs = solver_parameters.get('constrained_dofs', {})
        assert cstr_dofs
    else:
        raise NotImplementedError('For now')


def direct_solver(A, W, params):
    '''A direct solver with MUMPS'''
    if 'mumps' in lu_solver_methods().keys():
        solver = LUSolver(A, 'mumps')
    else:
        print 'MUMPS solver not found: using default'
        solver = LUSolver(A, 'default')
    # The system stays the same, only the rhs changes
    solver.parameters['reuse_factorization'] = True

    return solver

# -------------------------------------------------------------------

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from petsc4py import PETSc
    import numpy as np
    
    parameters['reorder_dofs_serial'] = False

    n = 512
    
    mesh = UnitSquareMesh(n, n)
    gamma_f = FacetFunction('size_t', mesh, 0)
    CompiledSubDomain('near(x[0]*(1-x[1])*(1-x[0]), 0.0)').mark(gamma_f, 1)

    V_elm = FiniteElement('Lagrange', triangle, 1)
    Q_elm = FiniteElement('Discontinuous Lagrange Trace', triangle, 0)

    V = FunctionSpace(mesh, V_elm)
    Q = FunctionSpace(mesh, Q_elm)

    u = TrialFunction(V)
    q = TestFunction(Q)
    
    W_elm = MixedElement([V_elm, Q_elm])
    W = FunctionSpace(mesh, W_elm)

    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)

    alpha = Constant(1.0)

    x, y = SpatialCoordinate(mesh)
    a = (inner(grad(u), grad(v))*dx + alpha*inner(u, v)*dx +
         inner(v, p)*ds + inner(u, q)*ds)
    L = inner(x+y+Constant(1), v)*dx + inner(x*y, q)*ds

    bc = DirichletBC(W.sub(1), Constant(0), gamma_f, 0)
    
    A, b = assemble_system(a, L, bc)

    solver = LUSolver('umfpack')
        
    w0 = Function(W)
    t = Timer('foo'); t.start()
    solver.solve(A, w0.vector(), b)
    print 'full', t.stop()

    t = Timer('bar'); t.start()
    # Dofs of W are first in dofmap
    # Not I want to compute perfmutation of Q dofs which puts the free
    # dofs first
    Q_dofs = W.sub(1).dofmap().dofs()
    Qcstr_dofs = bc.get_boundary_values().keys()
    Qfree_dofs = list(set(Q_dofs) - set(Qcstr_dofs))
    Qcstr_dofs = list(Qcstr_dofs)
    
    nVdofs = W.dim() - len(Q_dofs)
    nWfree_dofs = nVdofs + len(Qfree_dofs)
    
    Amat = as_backend_type(A).mat()
    bvec = as_backend_type(b).vec()

    # Diagonal to scale back cstr dofs
    diagonal = Amat.getDiagonal().getSubVector(PETSc.IS().createGeneral(Qcstr_dofs))
    diagonal = diagonal.array_r

    # Extract the free system
    free_dofs = np.r_[np.arange(nVdofs), Qfree_dofs]
    free_dofs = PETSc.IS().createGeneral(np.fromiter(free_dofs, dtype='int32'))

    Amat_red = Amat.getSubMatrix(free_dofs, free_dofs)
    bvec_red = bvec.getSubVector(free_dofs)

    # Reduce solve
    x_red = PETScVector(bvec_red.copy())
    # Solve
    solver = LUSolver('umfpack')
    solver.set_operator(PETScMatrix(Amat_red))
    solver.solve(x_red, PETScVector(bvec_red))
    x_red = x_red.vec()

    # Assign back
    w = Function(W)
    w_values = w.vector().get_local()
    w_values[np.arange(nVdofs)] = x_red.array[:nVdofs]
    w_values[Qfree_dofs] = x_red.array[nVdofs:]
    w_values[Qcstr_dofs] = bvec.array[Qcstr_dofs]/diagonal
    w.vector().set_local(w_values)
    w.vector().apply('insert')
    print 'reduced', t.stop()
    
    r = Function(W).vector()
    A.mult(w.vector(), r)
    r.axpy(-1, b)

    r0 = Function(W).vector()
    A.mult(w0.vector(), r0)
    r0.axpy(-1, b)
    
    print (w.vector() - w0.vector()).norm('linf')
    print W.dim(), x_red.size, (W.dim() - x_red.size)/float(W.dim())
    print r.norm('l2'), r0.norm('l2')        

    File('red.pvd') << w.split(deepcopy=True)[0]
