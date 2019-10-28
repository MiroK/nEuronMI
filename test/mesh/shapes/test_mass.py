from nEuronMI.mesh.shapes.gmsh_primitives import Sphere
import unittest, gmsh
from math import sqrt
import numpy as np


def as_array(x):
    return np.fromiter([x] if isinstance(x, (int, float)) else x, dtype=float)


def near(x, y, tol=1E-12):
    return np.linalg.norm(as_array(x)-as_array(y), np.inf) < tol


class TestSpere(unittest.TestCase):

    def test_full(self):
        model = gmsh.model
        fac = model.occ

        gmsh.initialize([])
        fac.addSphere(0, 0, 0, 1)
        fac.synchronize()

        x = model.occ.getCenterOfMass(2, 1)

        ball = Sphere([0, 0, 0], 1)

        y = ball._center_of_mass

        self.assertTrue(near(x, y))
