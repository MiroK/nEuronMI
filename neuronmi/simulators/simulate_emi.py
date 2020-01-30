from .solver.neuron_solver import neuron_solver
from .solver.aux import snap_to_nearest
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
                                 'start_time': 0.01,      # float: Start of stimulation in ms
                                 'stop_time': 1.0,        # float: Stop of stimulation in ms (it type is 'pulse')
                                 'syn_weight': 10.0,      # float: Synaptic weight in in mS/cm^2 (if 'type' is 'syn')
                                 'stim_current': 10,      # float: Stimulation current in nA (if 'type' is 'step'/'pulse')
                                 'position': [0, 0, 150], # array: 3D position of stimulation point in um.
                                 'length': 20,            # float: length of stimulated portion in um.
                                 'radius': 5,             # float: radius of stimulated area
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


def simulate_emi(mesh_folder, problem_params=None, u_probe_locations=None,
                 i_probe_locations=None, v_probe_locations=None,
                 save_simulation_output=True, save_format='pvd', save_folder=None, verbose=False):
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
    u_probe_locations: np.array
        Array of 3D points to measure potential
    i_probe_locations: np.array
        Array of 3D points to measure currents (if not on neuron surface the point is snapped to closest neuron point)
    v_probe_locations: np.array
        Array of 3D points to measure membrane potentials (if not on neuron surface the point is snapped
        to closest neuron point)
    save_simulation_output: bool
        If True, simulation output (u: potential, i: current, v: membrane potential) is saved
    save_format: str
        'pvd' or 'xdmf'
    save_folder: str
        Path to save folder. If None, an 'emi_simulation' folder is created in the mesh folder.
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

    if problem_params is None:
        problem_params = _default_problem_parameters

    #TODO yield membrane potential as well
    if save_folder is None:
        save_folder = mesh_folder / 'emi_simulation'
    else:
        save_folder = Path(save_folder)

    if save_simulation_output:
        if save_format == 'pvd':
            I_out = dolfin.File(str(save_folder / 'I.pvd'))
            u_out = dolfin.File(str(save_folder / 'u.pvd'))
        elif save_format == 'xdmf':
            I_out = dolfin.XDMFFile(str(save_folder / 'I.xdmf'))
            u_out = dolfin.XDMFFile(str(save_folder / 'u.xdmf'))
        else:
            raise AttributeError("'save_format' can be 'pvd' or 'xdmf'")

    dolfin.parameters['allow_extrapolation'] = True

    t_start = time.time()
    u_record = []
    i_record = []
    v_record = []
    I_proxy = None

    for (t, u, I) in neuron_solver(mesh_h5_path, emi_map, problem_params, scale_factor, verbose):
        if save_simulation_output:
            if save_format == 'pvd':
                I_out << I, t
                u_out << u, t
            elif save_format == 'xdmf':
                I_out.write(I, float(t))
                u_out.write(u, float(t))

        if I_proxy is None:
            I_proxy = snap_to_nearest(I)

        if u_probe_locations is not None:
            u_probe_t = np.zeros(len(u_probe_locations))
            for i, p in enumerate(u_probe_locations):
                u_probe_t[i] = u(p)
            u_record.append(u_probe_t)

        #TODO now I_proxy is in mAcm-2 --> multiply by facet area to obtain current!
        if i_probe_locations is not None:
            i_probe_t = np.zeros(len(i_probe_locations))
            for i, p in enumerate(i_probe_locations):
                i_probe_t[i] = I_proxy(p)
            i_record.append(i_probe_t)

        #TODO add membrane potential

    if u_probe_locations is not None:
        u_record = np.array(u_record)
    if i_probe_locations is not None:
        i_record = np.array(i_record)

    if save_simulation_output:
        print 'Results saved in ' + str(save_folder)
    print 'Elapsed time: ' + str(time.time() - t_start)

    return u_record, i_record
