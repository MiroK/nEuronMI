from .solver.neuron_solver import neuron_solver
import dolfin
from ..mesh.mesh_utils import EMIEntityMap, load_h5_mesh
from .solver.probing import get_geom_centers, Probe
import numpy as np
from copy import copy, deepcopy
from pathlib import Path
import time

_default_problem_parameters = {
    'neurons':  # List or dict. If list, length must be the same of number of neurons in mesh. If dict, the same
                # params are used for all neurons in the mesh
                {
                 'cond_int': 7,           # float: Intracellular conductivity in mS/cm^2
                 'Cm': 1.0,               # float: Membrane capacitance in uF/um^2
                 'models': {},            # dict: Models for neuron domains. Default: {'dendrite': 'pas', 'soma': 'hh',
                                          #                                             'axon': 'hh'}
                 'model_args': {},        # dict of tuples: Overwrite model default arguments.
                                          # E.g. ('dendrite', 'g_L'), ('soma', g_Na)'
                 'stimulation': {'type': 'syn',           # str: Stimulation type ('syn', 'step', 'pulse')
                                 'start_time': 0.1,       # float: Start of stimulation in ms
                                 'stop_time': 1.0,        # float: Stop of stimulation in ms (it type is 'pulse')
                                 'strength': 10.0,        # float: Stimulation strength in mS/cm^2
                                 'position': 150,         # float or array: position of stimulation in um. DEPRECATED
                                 'length': 20             # float: length od stimulated portion in um. DEPRECATED
                 }
    },
    'ext':     {
                 'cond_ext': 3,           # float: Extracellular conductivity: mS/cm^2
                 'insulated_bcs': []      # list: Insulated BC for bounding box. It can be: 'max_x', 'min_x', etc.
    },
    'probe':   {
                 'stimulated_sites': None,  # List or tuple: Stimulated electrodes (e.g. [contact_3, contact_22]
                 'type': None,            # str: Stimulation type ('step', 'pulse')
                 'start_time': 0.1,       # float: Start of stimulation in ms
                 'stop_time': 1.0,        # float: Stop of stimulation in ms (it type is 'pulse')
                 'current': 0             # float: Stimulation current in mA. If list, each value correspond to a stimulated
                                          #        site
    },
    'solver':  {
                 'dt_fem': 0.01,          # float: dt for fem solver in ms
                 'dt_ode': 0.01,          # float: dt for ode solver in ms
                 'sim_duration': 5,      # float: duration od the simulation in ms
    }
}


def get_default_emi_params():
    return deepcopy(_default_problem_parameters)


def simulate_emi(mesh_folder, problem_params=None, verbose=False):
    '''
    Simulates the 3d-3d EMI solution given a neuronmi-generated mesh (containing at least one neuron and optionally
    one probe).
    The simulation results (currents and potentials) are saved in the 'emi_sim' folder inside the mesh_folder.

    Parameters
    ----------
    mesh_folder: str or Path
        The path to the neuronmi-generated mesh containing .h5 and .json
    problem_params: dict
        Dictionary with simulation parameters. To retrieve default parameters, run **neuronmi.get_default_emi_params()**
    '''
    scale_factor = 1e-4
    mesh_folder = Path(mesh_folder)

    mesh_h5 = [f for f in mesh_folder.iterdir() if f.suffix == '.h5']
    mesh_json = [f for f in mesh_folder.iterdir() if f.suffix == '.json']

    if len(mesh_h5) != 1:
        raise ValueError('No or more than one .h5 mesh file found in %s' % mesh_folder)
    else:
        mesh_h5_path = mesh_h5[0]

    if len(mesh_json) != 1:
        raise ValueError('No or more than one .json mesh file found in %s' % mesh_folder)
    else:
        mesh_json_path = mesh_json[0]

    with mesh_json_path.open() as json_fp:
        emi_map = EMIEntityMap(json_fp=json_fp)

    # mesh, volumes, surfaces = load_h5_mesh(str(mesh_h5_path))
    #
    # probe_surfaces = emi_map.surface_physical_tags('probe')
    # contact_tags = [v for k, v in probe_surfaces.items() if 'contact_' in k]
    #
    # contact_centers = get_geom_centers(surfaces, contact_tags)
    # contact_centers = np.array(contact_centers) * scale_factor

    if problem_params is None:
        problem_params = _default_problem_parameters

    # TODO extract and save v_mem, v_probe, i_mem
    # I_out = dolfin.File(str(mesh_folder / 'emi_sim' / 'I.pvd'))
    # u_out = dolfin.File(str(mesh_folder / 'emi_sim' / 'u.pvd'))

    t_start = time.time()
    for (t, u, I) in neuron_solver(mesh_h5_path, emi_map, problem_params, scale_factor, verbose):
        yield t, u, I
        # I_out << I, t
        # u_out << u, t
    print 'Results saved in ' + str(mesh_folder / 'emi_sim')
    print 'Elapsed time: ' + str(time.time() - t_start)
