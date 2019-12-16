import neuronmi

p1 = neuronmi.get_neuron_params('bas')
p2 = neuronmi.get_neuron_params('tapered')
p3 = neuronmi.get_neuron_params('tapered')

p1['soma_y'] = -50
p3['soma_y'] = 50
p2['dend_len'] = 100

mesh3 = neuronmi.generate_mesh(neuron_type=['bas', 'tapered', 'tapered'], neuron_params=[p1, p2, p3])

