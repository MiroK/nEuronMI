import unittest, gmsh, subprocess, gmsh, json, os, time, glob
import numpy as np

from neuronmi.mesh.mesh_utils import *
from neuronmi.mesh.shapes import *


class TestEmiMesh(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        trash = ('h5', 'json', 'pvd', 'vtu', 'msh', 'geo_unrolled')
        map(os.remove, sum((glob.glob('*.%s' % ext) for ext in trash), []))
    
    # Saniyt checks for mesh
    def test_emi_mesh(self):
        gmsh.initialize()
        
        root = 'test_2neuron'
        msh_file = '%s.msh' % root
        
        # This gives course enough mesh that the solver runs fast
        box = Box(np.array([-60, -60, -100]), np.array([60, 60, 100]))
        
        neurons = [BallStickNeuron({'soma_x': 0, 'soma_y': 0, 'soma_z': 0,
                                    'soma_rad': 20, 'dend_len': 50, 'axon_len': 50,
                                    'dend_rad': 15, 'axon_rad': 10}),
                   TaperedNeuron({'soma_x': 30, 'soma_y': -30, 'soma_z': 0,
                                  'soma_rad': 20, 'dend_len': 20, 'axon_len': 20, 'axonh_len': 15, 'dendh_len': 15,
                                  'dend_rad': 10, 'axon_rad': 8, 'axonh_rad': 10, 'dendh_rad': 15})]
        
        probe = MicrowireProbe({'tip_x': 30, 'radius': 5, 'length': 800})
        
        # Coarse enough for tests
        size_params = {'DistMax': 20, 'DistMin': 10, 'LcMax': 40,
                       'neuron_LcMin': 6, 'probe_LcMin': 6}

        model = gmsh.model
        factory = model.occ

        model.add('Neuron')
        gmsh.option.setNumber('Mesh.PreserveNumberingMsh2', 1)
        gmsh.option.setNumber('Mesh.MshFileVersion', 2.2)
        
        # Add components to model
        model, mapping = build_EMI_geometry(model, box, neurons, probe=probe)
        
        with open('%s.json' % root, 'w') as out:
            mapping.dump(out)

        # Dump the mapping as json
        mesh_config_EMI_model(model, mapping, size_params)
        factory.synchronize()

        # # This is a way to store the geometry as geo file
        gmsh.write('%s.geo_unrolled' % root)
        # # 3d model
        model.mesh.generate(3)

        # # We have some mesh
        vtx_order, vtices, _ = model.mesh.getNodes()
        self.assertTrue(len(vtx_order))

        gmsh.write(msh_file)
        model.remove()        
        gmsh.finalize()

    def test_sandbox(self):
        # NOTE: load_mesh, msh_to_h5 or things to do with HDF5File cause
        # gmsh to fail on later tests. So these tests are isolated into
        # their own process 
        self.assertTrue(subprocess.call(['python ./sandbox/two_neurons.py'], shell=True))
