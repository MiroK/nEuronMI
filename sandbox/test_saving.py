import neuronmi
import numpy as np

test_folder = 'test_saving/'
microwire_probe = neuronmi.mesh.shapes.MicrowireProbe({'tip_x': 40})
mesh_with = neuronmi.generate_mesh(neurons='tapered', probe=microwire_probe, mesh_resolution=3, box_size=2,
                                   save_mesh_folder=test_folder)

# u_probe = [microwire_probe.get_electrode_centers(unit='cm')[0], [0, 0, 0]]

x_loc = np.linspace(-90, -10.1, 300)
y_loc = np.zeros_like(x_loc)
z_loc = np.zeros_like(x_loc)
u_probe = np.array([x_loc, y_loc, z_loc]).T * 1e-4

z_loc = np.arange(-30, 100)
y_loc = 3 * np.ones_like(z_loc)
x_loc = np.zeros_like(z_loc)
i_probe = np.array([[0, 0, 0], [0, 0, 150], [0, 0, -40]]) * 1e-4
#i_probe = np.array([x_loc, y_loc, z_loc]).T * 1e-4

u, i = neuronmi.simulate_emi(mesh_with, u_probe_locations=u_probe, i_probe_locations=i_probe, save_folder='test_saving2',
                             save_format='xdmf')

np.save(test_folder + 'u.npy', u)
np.save(test_folder + 'i.npy', i)

