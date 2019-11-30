from neuronmi.mesh.shapes.gmsh_primitives import *
import numpy as np
import unittest
import gmsh


class TestGmshPrimitives(unittest.TestCase):

    def test_inside(self):
        box = Box(np.array([0, 0, 0]), np.array([1, 1, 1]))
        self.assertTrue(box.contains(np.array([0.5, 0.5, 0.5]), 1E-13))

        sphere = Sphere(np.array([0, 0, 0]), 1)
        self.assertTrue(sphere.contains(np.array([0.5, 0.5, 0.5]), 1E-10))
        self.assertFalse(sphere.contains(np.array([1.5, 0.5, 0.5]), 1E-10))

        cyl = Cylinder(np.array([0, 0, 0]), np.array([1, 1, 1]), 1)
        self.assertFalse(cyl.contains(np.array([1.5, 1.5, 1.5]), 1E-10))
        self.assertTrue(cyl.contains(np.array([0.5, 0.5, 0.5]), 1E-10))

        cyl = Cone(np.array([0, 0, 0]), np.array([1, 1, 1]), 1, 2)
        self.assertFalse(cyl.contains(np.array([1.5, 1.5, 1.5]), 1E-10))
        self.assertTrue(cyl.contains(np.array([0.5, 0.5, 0.5]), 1E-10))

    def test_gmsh_primitives(self):
        shapes = [Box(np.array([0, 0, 0]), np.array([1, 1, 1])),
                  Sphere(np.array([0, 0, 0]), 1),
                  Cylinder(np.array([0, 0, 0]), np.array([1, 1, 1]), 1),
                  Cone(np.array([0, 0, 0]), np.array([1, 1, 1]), 1, 2)]

        gmsh.initialize()
        model = gmsh.model
        factory = model.occ

        for i, shape in enumerate(shapes):
            model.add(str(i))
            tag = shape.as_gmsh(model)
            factory.synchronize()

            x = model.occ.getCenterOfMass(3, tag)
            y = shape.center_of_mass
            self.assertTrue(np.linalg.norm(x-y) < 1E-13)

            model.mesh.generate(3)
            vtx_order, vtices, _ = model.mesh.getNodes()
            self.assertTrue(len(vtx_order))
            model.remove()
            
        gmsh.finalize()
