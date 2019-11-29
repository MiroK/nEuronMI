from neuronmi.mesh.shapes.utils import link_surfaces
from neuronmi.mesh.shapes import probe_list
from neuronmi.mesh.shapes import Box
from itertools import repeat
import numpy as np
import unittest
import gmsh


class TestMeshUtils(unittest.TestCase):
    def test_contact_find(self):
        # The idea here is that we should always be able to find the
        # contact surfaces of the standalone probe
        
        for Probe in probe_list.values():
            gmsh.initialize()

            probe = Probe()
            model = gmsh.model
            factory = model.occ
            # Probe can have more volumes
            volumes = probe.as_gmsh(model)
            volumes = list(zip(repeat(3), volumes))
            factory.synchronize()
            # Their bounding surface
            volumes_surfs = list(map(model.getBoundary, volumes))
            # A boundary can also contain curves - we ignore those; only keep 2d
            volumes_surfs = sum(([s[1] for s in ss if s[0] == 2] for ss in volumes_surfs), [])
            volumes_surfs = list(set(volumes_surfs))

            self.assertTrue(volumes_surfs)
        
            # At the moment its surfaces are all surfaces
            probe_surfaces = {}
            # We found all
            probe_surfaces = link_surfaces(model, volumes_surfs, probe, probe_surfaces,
                                           claim_last=False)

            # Find these
            is_contact = lambda s: 'contact_' in s
            want = set(filter(is_contact, probe._surfaces))
            have = set(filter(is_contact, probe_surfaces))

            self.assertTrue(want == have)
            # And we can make the mesh
            # model.mesh.generate(3)
            # vtx_order, vtices, _ = model.mesh.getNodes()
            # self.assertTrue(len(vtx_order))
        
            gmsh.finalize()
