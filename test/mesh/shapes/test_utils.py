from neuronmi.mesh.shapes.utils import *
import numpy as np
import unittest


class TestMeshUtils(unittest.TestCase):

    def test_find_first(self):
        my = find_first(4, range(19))
        truth = list(range(19)).index(4)

        self.assertTrue(my == truth)

    def test_circle_points(self):
        n = np.array([1, 1, 1])
        n = n/np.linalg.norm(n)
        c = np.array([0, 0, 0])
        r = 0.24
        pts = circle_points(c, r, a=n)
        # point - center is perp to axis; distance is correct
        for p in pts:
            # Orthogonalit
            self.assertTrue(abs(np.dot(p-c, n)) < 1E-13)
            self.assertTrue(abs(np.linalg.norm(p-c) - r) < 1E-13)

    def test_first(self):
        self.assertTrue(first((1, 0, 2)) == 1)

    def test_second(self):
        self.assertTrue(second((1, 0, 2)) == 0)

    def test(self):
        self.assertTrue(4 == find_first('f', 'xyXdfff'))
            
