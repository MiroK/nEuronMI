from neuronmi.mesh.shapes import neuron_list
import numpy as np
import unittest
import gmsh


class TestMeshUtils(unittest.TestCase):
    def test_create(self):
        for Neuron in neuron_list.values():
            neuron = Neuron()

            gmsh.initialize()

            model = gmsh.model
            factory = model.occ
            # Neuron defines one volume
            tag, surfs = neuron.as_gmsh(model)
            self.assertTrue(surfs)
            # At the moment its surfaces are all surfaces
            neuron_surfaces = {}
            match = neuron.link_surfaces(model, surfs, links=neuron_surfaces, tol=1E-10)
            # We found all
            self.assertFalse(surfs)
            self.assertTrue(set(neuron_surfaces.keys()) == set(neuron._surfaces.keys()))

            # And we can make the mesh
            # model.mesh.generate(3)
            # vtx_order, vtices, _ = model.mesh.getNodes()
            # self.assertTrue(len(vtx_order))
            
            gmsh.finalize()
