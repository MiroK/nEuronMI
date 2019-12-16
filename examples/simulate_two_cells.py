import neuronmi

p1 = neuronmi.get_neuron_params('bas')
p2 = neuronmi.get_neuron_params('bas')

p1['soma_y'] = -50

mesh_folder = neuronmi.generate_mesh(neuron_type=['bas', 'bas'], neuron_params=[p1, p2], probe_type=None,
                                     mesh_resolution=5, box_size=1)

neuron_params_0 = neuronmi.get_default_emi_params()['neurons']
neuron_params_1 = neuronmi.get_default_emi_params()['neurons']

neuron_params_1['stimulation']['strength'] = 0

params = neuronmi.get_default_emi_params()
params['neurons'] = [neuron_params_0, neuron_params_1]

neuronmi.simulate_emi(mesh_folder, params)
