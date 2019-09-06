import hashlib, time, os
from os.path import join
import subprocess, os
import numpy as np
from .shapes import BASNeuron, TaperedNeuron
from .shapes import MicrowireProbe, NeuronexusProbe, Neuropixels24Probe
from .shapes import neuron_list, probe_list


def generate_mesh(neuron_type='tapered', probe_type='neuronexus', mesh_resolution=2, box_size=2, neuron_params=None,
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
        Size of the boundig box. It can be 1, 2, 3, 4, 5, 6 (smaller to larger) or
        a dictionary with 'xlim', 'ylim', 'zlim' (scalar or vector of 2), which are the boundaries of the box
    neuron_params: dict
        Dictionary with neuron params: 'rad_soma', 'rad_dend', 'rad_axon', 'len_dend', 'len_axon'.
        If the 'neuron_type' is 'tapered', also 'rad_dend_base' and 'rad_axon_base'
    probe_params: dict
        Dictionary with probe params, including probe_tip and probe specific params (if any)
    save_mesh_path: str or Path
        The output path. If None, a 'mesh' folder is created in the current working directory.
    Returns
    -------
    mesh_h5: str
        Path to the h5-converted mesh, ready for simulation
    '''
    # generate
    # todo only generate 1 probe (with or without probe, with or without neuron)
    if isinstance(box_size, int):
        xlim, ylim, zlim = return_boxsizes(box_size)
    else:
        xlim = box_size['xlim']
        ylim = box_size['ylim']
        zlim = box_size['zlim']

    conv = 1e4

    if isinstance(mesh_resolution, int):
        mesh_resolution = return_coarseness(mesh_resolution)
    elif isinstance(mesh_resolution, dict):
        assert 'neuron' in mesh_resolution.keys()
        assert 'probe' in mesh_resolution.keys()
        assert 'rest' in mesh_resolution.keys()
    else:
        # set default here
        mesh_resolution = {'neuron': 3, 'probe': 6, 'rest': 9}

        # if neuron_params is None:
    #     geometrical_params = {'rad_soma': 10 * conv, 'rad_dend': 2.5 * conv, 'rad_axon': 1 * conv,
    #                           'length_dend': 400 * conv, 'length_axon': 200 * conv, 'rad_hilox_d': 4 * conv,
    #                           'length_hilox_d': 20 * conv, 'rad_hilox_a': 2 * conv, 'length_hilox_a': 10 * conv,
    #                           'dxp': dx * conv, 'dxn': dx * conv, 'dy': dy * conv, 'dz': dz * conv}
    # else:
    #     geometrical_params = neuron_params

    mesh_sizes = {'neuron': mesh_resolution['neuron'] * conv,
                  'probe': mesh_resolution['probe'] * conv,
                  'rest_mesh_size': mesh_resolution['rest'] * conv}

    # todo handle lists of neurons and neuron_params
    if neuron_type is not None and neuron_type in neuron_list.keys():
        neuron = neuron_list[neuron_type](neuron_params)

    if probe_type is not None and probe_type in probe_list.keys():
        probe = probe_list[probe_type](probe_params)

    # TODO make box

    # TODO assemble mesh
    # mesh = generate_mesh(neuron, probe, box, mesh_size)

    # TODO create folder and save files + params
    os.makedirs(save_mesh_folder)

    # return mesh.h5


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
    else:
        raise Exception('coarseness must be 00, 0, 1, 2, or 3')

    resolution = {'neuron': nmesh,
                  'probe': pmesh,
                  'rest': rmesh}
    return resolution


def return_boxsizes(box):
    if box == 1:
        dx = 80
        dy = 80
        dz = 80
    elif box == 2:
        dx = 100
        dy = 100
        dz = 100
    elif box == 3:
        dx = 120
        dy = 120
        dz = 120
    elif box == 4:
        dx = 160
        dy = 160
        dz = 160
    elif box == 5:
        dx = 200
        dy = 200
        dz = 200
    elif box == 6:
        dx = 300
        dy = 300
        dz = 300
    else:
        raise Exception('boxsize must be 1, 2, 3, 4, 5, or 6')

    return [-dx, dx], [-dy, dy], [-dz, dz]


def convert_msh2h5(msh_file, h5_file):
    '''Temporary version of convertin from msh to h5'''
    root, _ = os.path.splitext(msh_file)
    assert os.path.splitext(msh_file)[1] == '.msh'
    assert os.path.splitext(h5_file)[1] == '.h5'

    # Get the xml mesh
    xml_file = '.'.join([root, 'xml'])
    subprocess.call(['dolfin-convert %s %s' % (msh_file, xml_file)], shell=True)
    # Success?
    assert os.path.exists(xml_file)

    cmd = '''from dolfin import Mesh, HDF5File;\
             mesh=Mesh('%(xml_file)s');\
             assert mesh.topology().dim() == 3;\
             out=HDF5File(mesh.mpi_comm(), '%(h5_file)s', 'w');\
             out.write(mesh, 'mesh');''' % {'xml_file': xml_file,
                                            'h5_file': h5_file}

    for region in ('facet_region.xml', 'physical_region.xml'):
        name, _ = region.split('_')
        r_xml_file = '_'.join([root, region])
        if os.path.exists(r_xml_file):
            cmd_r = '''from dolfin import MeshFunction;\
                       f = MeshFunction('size_t', mesh, '%(r_xml_file)s');\
                       out.write(f, '%(name)s');\
                       ''' % {'r_xml_file': r_xml_file, 'name': name}

            cmd = ''.join([cmd, cmd_r])

    cmd = 'python -c "%s"' % cmd

    status = subprocess.call([cmd], shell=True)
    assert status == 0
    # Sucess?
    assert os.path.exists(h5_file)

    return True


def cleanup(files=None, exts=()):
    '''Get rid of xml'''
    if files is not None:
        return map(os.remove, files)
    else:
        files = filter(lambda f: any(map(f.endswith, exts)), os.listdir('.'))
        print('Removing', files)
        return cleanup(files)
