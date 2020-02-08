from neuronmi.simulators.simulate_emi import _default_problem_parameters
from neuronmi.simulators.solver.reduced_neuron_solver import neuron_solver

from .solver.aux import snap_to_nearest
import dolfin
from ..mesh.mesh_utils import EMIEntityMap, ReducedEMIEntityMap, load_h5_mesh
from .solver.probing import get_geom_centers, Probe
import numpy as np
from copy import copy, deepcopy
from pathlib import Path
import itertools
import time


def get_default_emi_params():
    return deepcopy(_default_problem_parameters)


def reduced_simulate_emi(mesh_folder, problem_params=None, u_probe_locations=None,
                         save_simulation_output=True, save_format='pvd', save_folder=None, verbose=False):
    '''
    Simulates the 3d-1d EMI solution given a neuronmi-generated mesh (containing at least one neuron and optionally
    one probe).
    The simulation results (currents and potentials) are saved in the 'emi_sim' folder inside the mesh_folder.

    Parameters
    ----------
    mesh_folder: str or Path
        The path to the neuronmi-generated mesh containing .h5 and .json
    problem_params: dict
        Dictionary with simulation parameters. To retrieve default parameters, run **neuronmi.get_default_emi_params()**
    u_probe_locations: list of lists
        the convention is that first list contains points where only extracecullar potential
        is sampled. The other lists are points for sampling intracellular potential on each
        neuron. Extracellular potential is sampled in these points as well.
    save_simulation_output: bool
        If True, simulation output (u: potential, i: current, v: membrane potential) is saved
    save_format: str
        'pvd' or 'xdmf'
    save_folder: str
        Path to save folder. If None, an 'emi_simulation' folder is created in the mesh folder.
    pde_formulation: str
        'mm' is m(ixed)m(ultidimensional)
        'pm' is p(rimal)m(ultidimensional)
        'ps' is p(rimal)s(ingledimensional)
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
        emi_map = ReducedEMIEntityMap(json_fp=json_fp)
        
    if problem_params is None:
        problem_params = _default_problem_parameters

    #TODO yield membrane potential as well
    if save_folder is None:
        save_folder = mesh_folder / 'emi_simulation'
    else:
        save_folder = Path(save_folder)


    dolfin.parameters['allow_extrapolation'] = True

    t_start = time.time()
    # We probe ith neuron at ith probe points but ue is probed
    # at all locations
    u_record, u_out = [], []

    for (t, u) in neuron_solver(mesh_h5_path, emi_map, problem_params, scale_factor, verbose):
        names = ['ue'] + ['ui%d' % i for i in range(len(u)-1)]
        if save_simulation_output:
            if not u_out:
                for name in names:
                    if save_format == 'pvd':
                        u_out.append(dolfin.File(str(save_folder / ('%s.pvd' % name))))
                    elif save_format == 'xdmf':
                        u_out.append(dolfin.XDMFFile(str(save_folder / ('%s.xdmf' % name))))
                    else:
                        raise AttributeError("'save_format' can be 'pvd' or 'xdmf'")
            else:
                for i, (name, ui) in enumerate(zip(names, u)):
                    if save_format == 'pvd':
                        u_out[i] << ui, t
                    else:
                        u_out[i].write(ui, float(t))

        if u_probe_locations is not None:
            ue_probes, ui_probes = u_probe_locations[0], u_probe_locations[1:]

            u_probe_t = []            
            # First eval extracellular
            ue = u[0]
            for p in ue_probes:
                u_probe_t.append(ue(p))
            # Noe extracecullular on internal points
            for p in itertools.chain(*ui_probes):
                u_probe_t.append(ue(p))
            # Now intracellular
            for pts, ui in zip(ui_probes, u[1:]):
                u_probe_t.extend(ui(p) for p in pts)
                
            u_record.append(np.array(u_probe_t))

    if u_probe_locations is not None:
        u_record = np.array(u_record)

    if save_simulation_output:
        print 'Results saved in ' + str(save_folder)
    print 'Elapsed time: ' + str(time.time() - t_start)

    return u_record
