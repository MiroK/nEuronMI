from neuronmi.simulators.solver.embedding import EmbeddedMesh
import dolfin as df
import numpy as np
import unittest


class TestCases(unittest.TestCase):
    
    def test(self):
        # Very basic
        mesh = df.UnitCubeMesh(10, 10, 10)

        f = df.MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
        chi = df.CompiledSubDomain('near(x[i], 0.5)', i=0) 

        for i in range(3):
            setattr(chi, 'i', i)
            chi.mark(f, i+1)

        mesh = EmbeddedMesh(f, [1, 2, 3])
        f = mesh.marking_function
        for i in range(3):
            mesh_i = df.SubMesh(mesh, f, i+1)
            x = mesh_i.coordinates()
            self.assertTrue(np.linalg.norm(x[:, i] - 0.5) < 1E-13)
            self.assertTrue(abs(xj) < 1E-13 for j, xj in enumerate(x.min(axis=0)) if j != i)
            self.assertTrue(abs(xj-1) < 1E-13 for j, xj in enumerate(x.min(axis=0)) if j != i)

