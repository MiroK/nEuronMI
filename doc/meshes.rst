Generating a mesh
=================

The :code:`mesh` module provides functions and utilities to ease the creation of meshes with neurons and neural devices.
The mesh is generated using `Gmsh <http://gmsh.info/>`_ as backend.

The user can create a mesh with the :code:`generate_mesh()` function.

.. code-block:: python

    mesh_folder = neuronmi.generate_mesh(neurons='bas', probe='microwire', mesh_resolution=3,
                                         box_size=3)

This snippet of code will generate a mesh with a ball-and-stick neuron (bas) and a microwire in the extracellular
space. The :code:`mesh_resolution` controls the resolution of the mesh (0 - fine resolution, 5 - coarse recolution).
The :code:`box_size` controls the size of the bounding box.

There are two kinds of neurons and three kinds of probes built-in.

Neurons:

- :code:`'bas'` :  ball-and-stick neuron
- :code:`'tapered'` :  similar to a ball-and-stick, but the connection between the soma and the dendrite/axon is tapered

Probes:

- :code:`'microwire'` : cylindrical probe sampling at its tip
- :code:`'neuronexus'` : Multi-Electrode Array from Neuronexus Technologies (A1x32-Poly3-5mm-25s-177-CM32)
- :code:`'neuropixels'` : Multi-electrode Array of `Neuropixels <https://www.neuropixels.org/>`_ technology

In order to retrieve the default parameters of a neuron or a probe, one can run:

.. code-block:: python

    neuron_params = neuronmi.get_neuron_params('bas')
    probe_params = neuronmi.get_probe_params('neuropixels')

Once the parameters are retrieved, they can be modified and used in the generate mesh function. In this example,
the position of the probe is modified.

.. code-block:: python

    probe_params = neuronmi.get_probe_params('microwire')
    probe_params['tip_x] = 30

    mesh_folder = neuronmi.generate_mesh(neurons='bas', probe='microwire', mesh_resolution=3,
                                         box_size=3, probe_params=probe_params)

While there can be at most one probe in the mesh, there can be multiple neurons. In order to simulate more than one
neuron, the user can use a list a the :code:`neurons` parameter. In this case, the :code:`neuron_params` must also be
a list:

.. code-block:: python

    neuron_params_1 = neuronmi.get_neuron_params('bas')
    neuron_params_2 = neuronmi.get_neuron_params('bas')

    # Displace two neurons
    neuron_params_1['soma_y'] = -20
    neuron_params_2['soma_y'] = 20

    mesh_folder = neuronmi.generate_mesh(neurons=['bas', 'bas'], probe='microwire', mesh_resolution=3,
                                         box_size=3, neuron_params=[neuron_params_1, neuron_params_2])


Finally, one can also instantiate neurons and probes outside and pass them to the :code:`generate_mesh()` function:

.. code-block:: python

    neuron = neuronmi.mesh.shapes.TaperedNeuron({'dend_len': 400, 'axon_len': 200})
    microwire_probe = neuronmi.mesh.shapes.MicrowireProbe({'tip_x': 30})
    mesh_folder = neuronmi.generate_mesh(neurons=neuron, probe=microwire_probe,
                                         mesh_resolution=3, box_size=3)

