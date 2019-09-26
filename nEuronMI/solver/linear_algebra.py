from itertools import chain
from petsc4py import PETSc
import numpy as np

from dolfin import *


def LinearSystemSolver(A, W, solver_parameters):
    '''
    This is a dispatches to solvers for EMI model. It returns something
    cable of solve(x, b) call.
    '''
    cstr_dofmap = solver_parameters.get('constrained_dofs', {})
    # Experimental solver which solves the system w/out the constrained dofs
    if cstr_dofmap:
        params = solver_parameters.copy()
        del params['constrained_dofs']
        
        info('Using reduced solver')
        return ReducedSolver(A, W, cstr_dofmap, params)
    # Unreduced
    if solver_parameters['linear_solver'] == 'direct':
        info('Using direct solver')
        return direct_solver(A, W, solver_parameters)
        
    else:
        raise NotImplementedError('For now')

    
class ReducedSolver(object):
    '''
    Assuming matrix A has been assembled with some artifial bcs which 
    constrain redundant degrees of freedom in the subspaces, this solver 
    solves Ax = b by forming first a reduced problem where the redundant 
    dofs are not present and then assembling x.
    '''
    def __init__(self, A, W, cstr_dofmap, params):
        # Getting free dofs for each of the subspaces
        free_dofs, cstr_dofs = [], []
        offsets = [0]  # Blocks for local numbering in the reduced system
        for i in range(W.num_sub_spaces()):
            all_sub_dofs = W.sub(i).dofmap().dofs()

            if i not in cstr_dofmap:
                free_dofs.append(list(all_sub_dofs))
                cstr_dofs.append([])
            else:
                free_dofs.append(list(set(all_sub_dofs) - cstr_dofmap[i]))
                cstr_dofs.append(list(cstr_dofmap[i]))
                
            offsets.append(offsets[-1] + len(free_dofs[-1]))

        # The reduced system is formed only once
        free_dofs_is = PETSc.IS().createGeneral(np.fromiter(chain(*free_dofs), dtype='int32'))
        Amat = as_backend_type(A).mat()
        Amat_reduced = Amat.getSubMatrix(free_dofs_is, free_dofs_is)
        A_reduced = PETScMatrix(Amat_reduced)

        # And we can also setup the solver which will fill the free guys
        self.solver = LinearSystemSolver(A_reduced, W, params)
        # The constrained guys are obtained by scaling the righ hand side
        # with the diagonal
        Adiag = Amat.getDiagonal().array_r
        Adiag = [Adiag[cstr] if cstr else [] for free, cstr in zip(free_dofs, cstr_dofs)]
        assert all(np.linalg.norm(diag) > 0 for diag in Adiag if len(diag) > 0), Adiag
        self.Adiag = Adiag
        
        # Reduction of the rhs
        self.free_is = free_dofs_is
        # Assembly back, The location of where to assign from reduced
        self.free_dofs = free_dofs
        self.cstr_dofs = cstr_dofs
        # Finally let's get how to extract from reduced
        self.offsets = offsets
        
    def solve(self, x, b):
        '''Each solve is reduction, reduced solve and assembly'''
        # Free solve
        b_reduced = PETScVector(as_backend_type(b).vec().getSubVector(self.free_is))
        x_reduced = b_reduced.copy()

        niters = self.solver.solve(x_reduced, b_reduced)
        x_reduced = as_backend_type(x_reduced).vec().array_r
        
        x_values = as_backend_type(x).vec().array_w
        b_values = as_backend_type(b).vec().array_r
        # Assembly back        
        for k, (free, cstr) in enumerate(zip(self.free_dofs, self.cstr_dofs)):

            x_values[free] = x_reduced[self.offsets[k]:self.offsets[k+1]]
            if cstr:
                x_values[cstr] = b_values[cstr]/self.Adiag[k]

        return niters


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
    print 'full', t.stop()

    t = Timer('bar'); t.start()
    solver = ReducedSolver(A, W,
                           cstr_dofmap={1: set(bcs[0].get_boundary_values().keys()),
                                        2: set(bcs[1].get_boundary_values().keys())},
                           params={'linear_solver': 'direct'})

    w = Function(W)
    solver.solve(w.vector(), b)

    red_size = solver.offsets[-1]

    print 'reduced', t.stop()

    r = Function(W).vector()
    A.mult(w.vector(), r)
    r.axpy(-1, b)

    r0 = Function(W).vector()
    A.mult(w0.vector(), r0)
    r0.axpy(-1, b)

    print (w.vector() - w0.vector()).norm('linf')/(w.vector().norm('linf'))
    print W.dim(), red_size, (W.dim() - red_size)/float(W.dim())
    print r.norm('l2'), r0.norm('l2')        

    File('red.pvd') << w.split(deepcopy=True)[0]

    # Single multiplier test
    if False:
        n = 256
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
        if False:
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

            red_size = x_red.size
        else:

            solver = ReducedSolver(A, W,
                                   cstr_dofmap={1: set(bc.get_boundary_values().keys())},
                                   params={'linear_solver': 'direct'})

            w = Function(W)
            solver.solve(w.vector(), b)

            red_size = solver.offsets[-1]

        print 'reduced', t.stop()

        r = Function(W).vector()
        A.mult(w.vector(), r)
        r.axpy(-1, b)

        r0 = Function(W).vector()
        A.mult(w0.vector(), r0)
        r0.axpy(-1, b)

        print (w.vector() - w0.vector()).norm('linf')/(w.vector().norm('linf'))
        print W.dim(), red_size, (W.dim() - red_size)/float(W.dim())
        print r.norm('l2'), r0.norm('l2')        

        File('red.pvd') << w.split(deepcopy=True)[0]