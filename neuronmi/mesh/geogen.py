from neuronmi.mesh.shapes import (Box, BallStickNeuron, TaperedNeuron,
                                  MicrowireProbe, #, NeuronexusProbe, Neuropixels24Probe
                                  neuron_list, probe_list)
from neuronmi.mesh.mesh_utils import build_EMI_geometry, mesh_config_EMI_model, msh_to_h5
import subprocess, os, sys, time
import numpy as np
import gmsh


def generate_mesh(neuron_type='bas', probe_type='microwire', mesh_resolution=2, box_size=2, neuron_params=None,
                  probe_params=None, save_mesh_folder=None):
    '''
    Parameters
    ----------
    neuron_type: str or None
        The neuron type (['bas' (ball-and-stick) or 'tapered' (tapered dendrite and axon)]
        If None, a mesh without neuron is generated.
    probe_type: str or None
        The probe type ('microwire', 'neuronexus', 'neuropixels-24')
        If None, a mesh without probe is generated.
    mesh_resolution: int or dict
        Resolution of the mesh. It can be 00, 0, 1, 2, 3 (less course to more coarse) or
        a dictionary with 'neuron', 'probe', 'rest' fields with cell size in um
    box_size: int or limits
        Size of the bounding box. It can be 1, 2, 3, 4, 5, 6 (smaller to larger) or
        a dictionary with 'xlim', 'ylim', 'zlim' (scalar or vector of 2), which are the boundaries of the box
    neuron_params: dict
        Dictionary with neuron params: 'rad_soma', 'rad_dend', 'rad_axon', 'len_dend', 'len_axon'.
        If the 'neuron_type' is 'tapered', also 'rad_dend_base' and 'rad_axon_base'
    probe_params: dict
        Dictionary with probe params, including probe_tip and probe specific params (if any)
    save_mesh_folder: str
        The output path. If None, a 'mesh' folder is created in the current working directory.
    Returns
    -------
    save_mesh_folder: str
        Path to the mesh folder, ready for simulation
    '''
    # todo only generate 1 probe (with or without probe, with or without neuron)
    if isinstance(box_size, int):
        xlim, ylim, zlim = return_boxsizes(box_size)
    else:
        xlim = box_size['xlim']
        ylim = box_size['ylim']
        zlim = box_size['zlim']
    if np.array(xlim).size == 1:
        xlim = np.array([xlim, xlim])
    if np.array(ylim).size == 1:
        ylim = np.array([ylim, ylim])
    if np.array(zlim).size == 1:
        zlim = np.array([zlim, zlim])

    box = Box(np.array([xlim[0], ylim[0], zlim[0]]), np.array([xlim[1], ylim[1], zlim[1]]))
    # box = Box(np.array([-100, -100, -100]), np.array([200, 200, 500]))

    if isinstance(mesh_resolution, int):
        mesh_resolution = return_coarseness(mesh_resolution)
    elif isinstance(mesh_resolution, dict):
        assert 'neuron' in mesh_resolution.keys()
        assert 'probe' in mesh_resolution.keys()
        assert 'ext' in mesh_resolution.keys()
    else:
        # set default here
        mesh_resolution = {'neuron': 3, 'probe': 6, 'ext': 9}

    # load correct neuron and probe
    # todo handle lists of neurons and neuron_params
    if neuron_type is not None and neuron_type in neuron_list.keys():
        neuron = neuron_list[neuron_type](neuron_params)
        neuron_str = neuron_type
    else:
        neuron = None
        neuron_str = 'noneuron'

    if probe_type is not None and probe_type in probe_list.keys():
        probe = probe_list[probe_type](probe_params)
        probe_str = probe_type
    else:
        probe = None
        probe_str = 'noprobe'

    mesh_sizes = {'neuron': mesh_resolution['neuron'],
                  'probe': mesh_resolution['probe'],
                  'ext': mesh_resolution['ext']}

    # Coarse enough for tests
    size_params = {'DistMax': 20, 'DistMin': 10, 'LcMax': mesh_sizes['ext'],
                   'neuron_LcMin': mesh_sizes['neuron'], 'probe_LcMin': mesh_sizes['probe']}

    if save_mesh_folder is None:
        mesh_name = 'mesh_%s_%s_%s' % (neuron_str, probe_str, time.strftime("%d-%m-%Y_%H-%M"))
        save_mesh_folder = mesh_name
    else:
        mesh_name = save_mesh_folder
        save_mesh_folder = save_mesh_folder

    if not os.path.isdir(save_mesh_folder):
        os.makedirs(save_mesh_folder)  # FIXME: , exist_ok=True)

    # Components
    model = gmsh.model
    factory = model.occ
    # You can pass -clscale 0.25 (to do global refinement)
    # or -format msh2            (to control output format of gmsh)
    args = sys.argv + ['-format', 'msh2', '-clscale', '0.5']  # Dolfin convert handles only this
    gmsh.initialize(args)

    gmsh.option.setNumber("General.Terminal", 1)

    # # Add components to model
    model, mapping = build_EMI_geometry(model, box, neuron, probe) #, mapping
    # # Config fields and dump the mapping as json
    mesh_config_EMI_model(model, mapping, size_params)
    json_file = os.path.join(save_mesh_folder, '%s.json' % mesh_name)
    with open(json_file, 'w') as out:
        mapping.dump(out)

    factory.synchronize()
    # This is a way to store the geometry as geo file
    geo_unrolled_file = os.path.join(save_mesh_folder, '%s.geo_unrolled' % mesh_name)
    gmsh.write(str(geo_unrolled_file))
    # gmsh.fltk.initialize()
    # gmsh.fltk.run()
    # 3d model
    model.mesh.generate(3)
    # Native optimization
    model.mesh.optimize('')
    msh_file = os.path.join(save_mesh_folder, '%s.msh' % mesh_name)
    gmsh.write(str(msh_file))
    gmsh.finalize()

    # Convert
    h5_file = os.path.join(save_mesh_folder, '%s.h5' % mesh_name)
    msh_to_h5(msh_file, str(h5_file))

    return save_mesh_folder

def return_coarseness(coarse):
    if coarse == 00:
        nmesh = 2
        pmesh = 3
        rmesh = 5
    elif coarse == 0:
        nmesh = 2
        pmesh = 5
        rmesh = 7.5
    elif coarse == 1:
        nmesh = 3
        pmesh = 6
        rmesh = 9
    elif coarse == 2:
        nmesh = 4
        pmesh = 8
        rmesh = 12
    elif coarse == 3:
        nmesh = 4
        pmesh = 10
        rmesh = 15
    elif coarse == 4:
        nmesh = 10
        pmesh = 15
        rmesh = 20
    elif coarse == 5:
        nmesh = 15
        pmesh = 20
        rmesh = 30
    else:
        raise Exception('coarseness must be 00, 0, 1, 2, or 3')

    resolution = {'neuron': nmesh,
                  'probe': pmesh,
                  'ext': rmesh}
    return resolution


def return_boxsizes(box):
    if box == 1:
        dx = 80
        dy = 80
        dz = 220
    elif box == 2:
        dx = 100
        dy = 100
        dz = 240
    elif box == 3:
        dx = 120
        dy = 120
        dz = 260
    elif box == 4:
        dx = 160
        dy = 160
        dz = 280
    elif box == 5:
        dx = 200
        dy = 200
        dz = 300
    elif box == 6:
        dx = 300
        dy = 300
        dz = 500
    else:
        raise Exception('boxsize must be 1, 2, 3, 4, 5, or 6')

    return np.array([-dx, dx]), np.array([-dy, dy]), np.array([-dz, dz])

