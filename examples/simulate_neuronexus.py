import neuronmi

mesh_folder = neuronmi.generate_mesh(neuron_type='tapered', probe_type='neuronexus', box_size=4, mesh_resolution=4)
neuronmi.simulate_emi(mesh_folder)
