import neuronmi
import numpy as np

test_folder = 'test_saving/'
microwire_probe = neuronmi.mesh.shapes.MicrowireProbe({'tip_x': 40})
mesh_with = neuronmi.generate_mesh(neurons='tapered', probe=microwire_probe, mesh_resolution=5, box_size=2,
                                   save_mesh_folder=test_folder)

u_probe = [microwire_probe.get_electrode_centers(unit='cm')[0], [0, 0, 0]]
i_probe = np.array([[10, 0, 0], [2, 0, 150], [1, 0, -40]]) * 1e-4

u, i = neuronmi.simulate_emi(mesh_with, u_probe_locations=u_probe, i_probe_locations=i_probe)

np.save(test_folder + 'u.npy', u)
np.save(test_folder + 'i.npy', i)

