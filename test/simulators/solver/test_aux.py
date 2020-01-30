from neuronmi.simulators.solver.aux import *
#from neuronmi.simulators.solver.embedding import EmbeddedMesh
import dolfin as df
import numpy as np
import unittest


class TestCases(unittest.TestCase):
    
    def test_snap_nearest(self):
        mesh = df.UnitCubeMesh(10, 10, 10)
        mesh = df.BoundaryMesh(mesh, 'exterior')
        V = df.FunctionSpace(mesh, 'CG', 2)
        f = df.interpolate(df.Expression('sin(x[0]+x[1]+x[2])', degree=2), V)
        
        # A proxy expression whose action at x eval f at closest point dof to x
        f_ = snap_to_nearest(f)
        self.assertTrue(abs(f_(1., 1., 1.) - f(1., 1., 1.)) < 1E-13)
        self.assertTrue(abs(f_(1.1, 1.1, 1.1) - f(1., 1., 1.)) < 1E-13)

    def test_subdomain_bbox_2d(self):
        mesh = df.UnitSquareMesh(10, 10)
        cell_f = df.MeshFunction('size_t', mesh, 2, 0)
        df.CompiledSubDomain('x[0] > 0.5 - DOLFIN_EPS && x[1] > 0.5 - DOLFIN_EPS').mark(cell_f, 1)
        
        self.assertTrue(subdomain_bbox(cell_f, 1) == [(0.5, 1.0), (0.5, 1.0)])

    def test_subdomain_bbox_3d(self):
        mesh = df.UnitCubeMesh(10, 10, 10)
        cell_f = df.MeshFunction('size_t', mesh, 3, 0)
        df.CompiledSubDomain('x[0] > 0.5 - DOLFIN_EPS && x[1] > 0.5 - DOLFIN_EPS').mark(cell_f, 1)

        self.assertTrue(subdomain_bbox(cell_f, 1) == [(0.5, 1.0), (0.5, 1.0), (0.0, 1.0)])

    def test_subdomain_bbox_3d_multi(self):
        mesh = df.UnitSquareMesh(10, 10)
        cell_f = df.MeshFunction('size_t', mesh, 2, 0)
        df.CompiledSubDomain('x[0] > 0.5 - DOLFIN_EPS && x[1] > 0.5 - DOLFIN_EPS').mark(cell_f, 1)
        df.CompiledSubDomain('x[0] < 0.5 + DOLFIN_EPS && x[1] > 0.5 - DOLFIN_EPS').mark(cell_f, 2)
        df.CompiledSubDomain('x[0] < 0.5 + DOLFIN_EPS && x[1] < 0.5 + DOLFIN_EPS').mark(cell_f, 3)

        self.assertTrue(subdomain_bbox(cell_f) == [(0.0, 1.0), (0.0, 1.0)])
        self.assertTrue(subdomain_bbox(cell_f, (1, 2)) == [(0.0, 1.0), (0.5, 1.0)])
        self.assertTrue(subdomain_bbox(cell_f, (3, 2)) == [(0.0, 0.5), (0.0, 1.0)])
        self.assertTrue(subdomain_bbox(cell_f, (1, 0)) == [(0.5, 1.0), (0.0, 1.0)])
        self.assertTrue(subdomain_bbox(cell_f, (1, 3)) == [(0.0, 1.0), (0.0, 1.0)])

    def test_closest_point(self):
        mesh = df.UnitCubeMesh(10, 10, 10)
        facet_f = df.MeshFunction('size_t', mesh, 2, 0)
        df.CompiledSubDomain('near(x[0], 0)').mark(facet_f, 1)
        df.CompiledSubDomain('near(x[2], 0.5)').mark(facet_f, 2)

        x = np.array([1, 1., 1.])
        entity = closest_entity(x, facet_f)
        x0 = entity.midpoint().array()[:3]

        print x0

        #f = point_source(entity, A=df.Expression('3', degree=1))
        #self.assertTrue(abs(f(x0) - 3) < 1E-15),
        #self.assertTrue(abs(f(x0 + 1E-9*np.ones(3))) < 1E-15)

    def test_surface_normal(self):
        mesh = df.UnitCubeMesh(5, 5, 5)
        bdries = df.MeshFunction('size_t', mesh, 2, 0)
        df.CompiledSubDomain('near(x[0], 0)').mark(bdries, 1)
        
        n = surface_normal(1, bdries, [0.5, 0.5, 0.5])
        n = np.array([-1, 0, 0])
        self.assertTrue(np.linalg.norm(n - np.array([-1, 0, 0])) < 1E-13)

    def test_site_current(self):
        mesh = df.UnitCubeMesh(5, 5, 5)
        bdries = df.MeshFunction('size_t', mesh, 2, 0)
        df.CompiledSubDomain('near(x[0], 0)').mark(bdries, 1)
        
        n = surface_normal(1, bdries, [0.5, 0.5, 0.5])
        
        mag = df.Expression('x[0]*t', t=0, degree=1)
        I = SiteCurrent(I=mag, n=n, degree=1)
        for t in range(1, 10):
            I.t = t
            self.assertTrue(np.linalg.norm(I(1, 0, 0) - t*n) < 1E-13)
            self.assertTrue(np.linalg.norm(I(0, 0, 0)) < 1E-13)

# ---------------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
