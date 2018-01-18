from dolfin import *


def LinearSystemSolver(A, W, solver_parameters):
    '''
    This is a dispatches to solvers for EMI model. It returns something
    cable of solve(x, b) call.
    '''
    if solver_parameters['linear_solver'] == 'direct':
        return direct_solver(A, W, solver_parameters)
    else:
        raise NotImplementedError('For now')


def direct_solver(A, W, params):
    '''A direct solver with MUMPS'''
    solver = LUSolver(A, 'mumps')
    # The system stays the same, only the rhs changes
    solver.parameters['reuse_factorization'] = True

    return solver
