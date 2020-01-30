import neuronmi
import numpy as np

test_folder = 'test_stim/'
microwire_probe = neuronmi.mesh.shapes.MicrowireProbe({'tip_x': 40})
mesh_with = neuronmi.generate_mesh(neurons='tapered', probe=microwire_probe, mesh_resolution=5, box_size=2,
                                   save_mesh_folder=test_folder)

u_probe = [microwire_probe.get_electrode_centers(unit='cm')[0], [0, 0, 0], [0, 0, 150*1e-4]]

# x_loc = np.linspace(-90, -10.1, 300)
# y_loc = np.zeros_like(x_loc)
# z_loc = np.zeros_like(x_loc)
# u_probe = np.array([x_loc, y_loc, z_loc]).T * 1e-4

z_loc = np.arange(-30, 100)
y_loc = 3 * np.ones_like(z_loc)
x_loc = np.zeros_like(z_loc)
i_probe = np.array([[0, 0, 0], [0, 0, 150], [0, 0, -40]])

v_probe = np.array([[0, 0, 0], [0, 0, 150], [0, 0, -40]])

params = neuronmi.get_default_emi_params()

params['neurons']["stimulation"]["type"] = "step"
params['neurons']["stimulation"]["stim_current"] = 0.1  # nA
params['neurons']["stimulation"]["syn_weight"] = 10  # mS/um2
params['neurons']["stimulation"]["position"] = [0, 0, 30]
params['neurons']["stimulation"]["length"] = None
params['neurons']["stimulation"]["radius"] = 5
params['solver']['sim_duration'] = 0.5

u, i, v = neuronmi.simulate_emi(mesh_with, u_probe_locations=u_probe, i_probe_locations=i_probe, v_probe_locations=v_probe,
                             save_folder='test_stim', problem_params=params,
                             save_format='xdmf')

np.save(test_folder + 'u.npy', u)
np.save(test_folder + 'v.npy', v)
np.save(test_folder + 'i.npy', i)

