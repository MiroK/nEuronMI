import numpy as np
from neuronmi.mesh.shapes import (Box, Neuron, Probe, neuron_list, probe_list)
from neuronmi.mesh.mesh_utils import build_EMI_geometry, mesh_config_EMI_model, msh_to_h5
import subprocess, os, sys, time
from pathlib import Path
from copy import copy
import gmsh


def get_available_neurons():
    return neuron_list.keys()


def get_neuron_params(neuron_name):
    if neuron_name in neuron_list.keys():
        return copy(neuron_list[neuron_name]._defaults)


def get_available_probes():
    return probe_list.keys()


def get_probe_params(probe_name):
    if probe_name in probe_list.keys():
        return copy(probe_list[probe_name]._defaults)


def generate_mesh(neurons='bas', probe='microwire', mesh_resolution=2, box_size=2, neuron_params=None,
                  probe_params=None, save_mesh_folder=None):
    '''
    Generate mesh with neurons and probes.

    Parameters
    ----------
    neurons: str, list, Neuron object, or None
        The neuron type (['bas' (ball-and-stick) or 'tapered' (tapered dendrite and axon)]).
        If list (of str or Neuron objects), multiple neurons are inserted in the mesh.
        If Neuron object instantiated outside the function, the neuron is inserted in the mesh as is.
        If None, a mesh without neuron is generated.
    probe: str, Probe object, or None
        The probe type ('microwire', 'neuronexus', 'neuropixels')
        If Probe object instantiated outside the function, the probe is inserted in the mesh as is.
        If None, a mesh without probe is generated.
    mesh_resolution: int or dict
        Resolution of the mesh. It can be 0, 1, 2, 3, 4, 5 (less course to more coarse) or
        a dictionary with 'neuron', 'probe', 'ext' fields with cell size in um
    box_size: int or limits
        Size of the bounding box. It can be 1, 2, 3, 4, 5, 6 (smaller to larger) or
        a dictionary with 'xlim', 'ylim', 'zlim' (vector of 2), which are the boundaries of the box
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
    import numpy as np

    if isinstance(box_size, int):
        xlim, ylim, zlim = return_boxsizes(box_size)
    else:
        xlim = box_size['xlim']
        ylim = box_size['ylim']
        zlim = box_size['zlim']
    assert len(xlim) == 2 and len(ylim) == 2 and len(zlim) == 2, 'Box dimensions should be 2-dimensional (min-max)'

    box = Box(np.array([xlim[0], ylim[0], zlim[0]]), np.array([xlim[1], ylim[1], zlim[1]]))

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

    neuron_str = None
    if neurons is not None:
        if isinstance(neurons, str):
            if neurons in neuron_list.keys():
                neuron_objects = neuron_list[neurons](neuron_params)
                neuron_str = neurons
            else:
                raise AttributeError("'neurons' not in %s" % neuron_list.keys())
        elif isinstance(neurons, Neuron):
            neuron_objects = neurons
            neuron_str = neurons.get_neuron_type()
        elif isinstance(neurons, list):
            if isinstance(neurons[0], str):
                assert neuron_params is not None and len(neuron_params) == len(neurons), "For a list of neurons, " \
                                                                                         "provide a list of neuron_" \
                                                                                         "params!"
                neuron_objects = []
                neuron_str = ''
                for i, (nt, np) in enumerate(zip(neurons, neuron_params)):
                    neuron_objects.append(neuron_list[nt](np))
                    if i == 0:
                        neuron_str += '%s' % nt
                    else:
                        neuron_str += '-%s' % nt
            elif isinstance(neurons[0], Neuron):
                neuron_objects = neurons
                neuron_str = ''
                for i, nt in enumerate(neurons):
                    if i == 0:
                        neuron_str += '%s' % nt.get_neuron_type()
                    else:
                        neuron_str += '-%s' % nt.get_neuron_type()
        else:
            raise AttributeError("'neurons' can be str, Neuron, or list")
    else:
        neuron_objects = None
        neuron_str = 'noneuron'

    if probe is not None:
        if isinstance(probe, str):
            if probe in probe_list.keys():
                probe_object = probe_list[probe](probe_params)
                probe_str = probe
            else:
                raise AttributeError("'probe' not in %s" % probe_list.keys())
        elif isinstance(probe, Probe):
            probe_object = probe
            probe_str = probe.get_probe_type()
        else:
            raise AttributeError("'probe' can be str or Probe")
    else:
        probe_object = None
        probe_str = 'noprobe'

    mesh_sizes = {'neuron': mesh_resolution['neuron'],
                  'probe': mesh_resolution['probe'],
                  'ext': mesh_resolution['ext']}

    # Coarse enough for tests
    size_params = {'DistMax': 20, 'DistMin': 10, 'LcMax': mesh_sizes['ext'],
                   'neuron_LcMin': mesh_sizes['neuron'], 'probe_LcMin': mesh_sizes['probe']}

    mesh_name = 'mesh_%s_%s_%s' % (neuron_str, probe_str, time.strftime("%d-%m-%Y_%H-%M"))
    if save_mesh_folder is None:
        save_mesh_folder = mesh_name
    else:
        save_mesh_folder = str(Path(save_mesh_folder) / mesh_name)

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
    model, mapping = build_EMI_geometry(model, box, neuron_objects, probe_object)  # , mapping
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
    if coarse == 0:
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
        raise Exception('coarseness must be 0, 1, 2, or 3')

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
