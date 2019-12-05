from neuronmi.simulators.solver.embedding import EmbeddedMesh
from neuronmi.simulators.solver.probing import *
import dolfin as df
import numpy as np
import unittest


class TestCases(unittest.TestCase):

    def test_centers(self):
        mesh = df.UnitCubeMesh(4, 4, 4)
        facet_f = df.MeshFunction('size_t', mesh, 2, 0)
        df.CompiledSubDomain('near(x[0], 0)').mark(facet_f, 1)
        df.CompiledSubDomain('near(x[0], 1)').mark(facet_f, 2)

        x = np.array(get_geom_centers(facet_f, (1, 2)))
        y = np.array([[0, 0.5, 0.5],
                      [1, 0.5, 0.5]])

        self.assertTrue(np.linalg.norm(x - y) < 1E-13)

    def test_probe(self):
        mesh = df.UnitCubeMesh(4, 4, 4)
        f = df.Expression('t*t*x[0] + t*x[1] + x[2]', t=0, degree=1)

        V = df.FunctionSpace(mesh, 'CG', 1)
        fh = df.Function(V)

        locations = np.array([[0.2, 0.3, 0.4], [0.1, 0.5, 0.2]])
        
        probe = Probe(fh, locations)
        for t in range(1, 5):
            f.t = t
            fh.vector()[:] = df.interpolate(f, V).vector()

            probe.probe(t)
            values = probe.data[-1][1:]
            self.assertTrue(all(abs(f(loc) - v) < 1E-13 for loc, v in zip(locations, values)))

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
