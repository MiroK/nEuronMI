import neuronmi
import numpy as np

microwire = False
neuronexus = True
neurpixels = False

if microwire:
    # Microwire
    microwire_folder = 'probe_effect/microwire/'
    microwire_probe = neuronmi.mesh.shapes.MicrowireProbe()
    mesh_with = neuronmi.generate_mesh(neurons='tapered', probe=microwire_probe, mesh_resolution=4, box_size=4,
                                       save_mesh_folder=microwire_folder)

    mesh_without = neuronmi.generate_mesh(neurons='tapered', probe=None, mesh_resolution=4, box_size=4,
                                          save_mesh_folder=microwire_folder)

    centers_microwire = microwire_probe.get_electrode_centers(unit='cm')

    u_with, _ = neuronmi.simulate_emi(mesh_with, u_probe_locations=centers)
    u_without, _ = neuronmi.simulate_emi(mesh_with, u_probe_locations=centers)

    np.save(microwire_folder + 'u_with.npy', u_with)
    np.save(microwire_folder + 'u_without.npy', u_without)
    np.save(microwire_folder + 'centers.npy', centers_neuronexus)

if neuronexus:
    # Neuronexus
    neuronexus_folder = 'probe_effect/neuronexus/'
    neuronexus_probe = neuronmi.mesh.shapes.NeuronexusProbe()
    mesh_with = neuronmi.generate_mesh(neurons='tapered', probe=neuronexus_probe, mesh_resolution=4, box_size=4,
                                       save_mesh_folder=neuronexus_folder)

    mesh_without = neuronmi.generate_mesh(neurons='tapered', probe=None, mesh_resolution=4, box_size=4,
                                          save_mesh_folder=neuronexus_folder)

    centers_neuronexus = neuronexus_probe.get_electrode_centers(unit='cm')

    u_with, _ = neuronmi.simulate_emi(mesh_with, u_probe_locations=centers)
    u_without, _ = neuronmi.simulate_emi(mesh_with, u_probe_locations=centers)

    np.save(neuronexus_folder + 'u_with.npy', u_with)
    np.save(neuronexus_folder + 'u_without.npy', u_without)
    np.save(neuronexus_folder + 'centers.npy', centers_neuronexus)

if neuropixels:
    # Neuropixels
    neuropixels_folder = 'probe_effect/neuropixels/'
    neuropixels_probe = neuronmi.mesh.shapes.NeuropixelsProbe()
    mesh_with = neuronmi.generate_mesh(neurons='tapered', probe=neuropixels_probe, mesh_resolution=4, box_size=4,
                                       save_mesh_folder=neuropixels_folder)

    mesh_without = neuronmi.generate_mesh(neurons='tapered', probe=None, mesh_resolution=4, box_size=4,
                                          save_mesh_folder=neuropixels_folder)

    centers_neuropixels = neuropixels_probe.get_electrode_centers(unit='cm')

    u_with, _ = neuronmi.simulate_emi(mesh_with, u_probe_locations=centers)
    u_without, _ = neuronmi.simulate_emi(mesh_with, u_probe_locations=centers)

    np.save(neuropixels_folder + 'u_with.npy', u_with)
    np.save(neuropixels_folder + 'u_without.npy', u_without)
    np.save(neuropixels_folder + 'centers.npy', centers_neuropixels)