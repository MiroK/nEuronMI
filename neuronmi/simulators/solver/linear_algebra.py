from itertools import chain
from petsc4py import PETSc
import numpy as np

from dolfin import *


def LinearSystemSolver(A, W, solver_parameters):
    '''
    This is a dispatches to solvers for EMI model. It returns something
    cable of solve(x, b) call.
    '''
    return direct_solver(A, W, solver_parameters)


def direct_solver(A, W, params):
    '''A direct solver with MUMPS'''
    if 'mumps' in lu_solver_methods().keys():
        solver = LUSolver(A, 'mumps')
    else:
        print('MUMPS solver not found: using default')
        solver = LUSolver(A, 'default')
    # The system stays the same, only the rhs changes

    return solver

# -------------------------------------------------------------------

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from petsc4py import PETSc
    import numpy as np
    
    parameters['reorder_dofs_serial'] = False

    n = 256
    mesh = UnitSquareMesh(n, n)
    
    gamma_f0 = FacetFunction('size_t', mesh, 0)
    CompiledSubDomain('near(x[0], 0.0)').mark(gamma_f0, 1)

    gamma_f1 = FacetFunction('size_t', mesh, 0)    
    CompiledSubDomain('near(x[0], 1.0)').mark(gamma_f1, 1)

    V_elm = FiniteElement('Lagrange', triangle, 1)
    Q_elm = FiniteElement('Discontinuous Lagrange Trace', triangle, 0)

    W_elm = MixedElement([V_elm, Q_elm, Q_elm])
    W = FunctionSpace(mesh, W_elm)

    u, p0, p1 = TrialFunctions(W)
    v, q0, q1 = TestFunctions(W)

    alpha = Constant(1.0)

    x, y = SpatialCoordinate(mesh)
    a = (inner(grad(u), grad(v))*dx + alpha*inner(u, v)*dx +
         inner(v, p0)*ds + inner(u, q0)*ds +
         inner(v, p1)*ds + inner(u, q1)*ds)
    
    L = inner(x+y+Constant(1), v)*dx + inner(y, q0)*ds + inner(-y, q1)*ds

    bcs = [DirichletBC(W.sub(1), Constant(0), gamma_f0, 0),
           DirichletBC(W.sub(2), Constant(0), gamma_f1, 0)]

    A, b = assemble_system(a, L, bcs)

    solver = LUSolver('umfpack')

    w0 = Function(W)
    t = Timer('foo'); t.start()
    solver.solve(A, w0.vector(), b)
    print('full', t.stop())
