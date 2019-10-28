from neuronmi.mesh.shapes.utils import *
import unittest


class TestMeshUtils(unittest.TestCase):

    def test_find_first(self):
        my = find_first(4, range(19))
        truth = list(range(19)).index(4)

        self.assertTrue(my == truth)
