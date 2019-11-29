from itertools import chain
from petsc4py import PETSc
import numpy as np

from dolfin import *


def LinearSystemSolver(A, W, solver_parameters=None):
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
