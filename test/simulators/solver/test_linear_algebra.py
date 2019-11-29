from neuronmi.simulators.solver.linear_algebra import *
import dolfin as df
import numpy as np
import unittest


class TestCases(unittest.TestCase):

    def test(self):
        # This doesn't solve any relevant problem
        n = 32
        mesh = UnitSquareMesh(n, n)
        fdim = mesh.topology().dim() - 1
        
        gamma_f0 = MeshFunction('size_t', mesh, fdim, 0)
        CompiledSubDomain('near(x[0], 0.0)').mark(gamma_f0, 1)
        
        gamma_f1 = MeshFunction('size_t', mesh, fdim, 0)    
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

        solver = LinearSystemSolver(A, W)

        w0 = Function(W)
        t = Timer('foo'); t.start()
        niter = solver.solve(A, w0.vector(), b)

        self.assertTrue(niter == 1)
