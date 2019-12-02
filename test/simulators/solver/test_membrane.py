from neuronmi.simulators.solver.transferring import SubMeshTransfer
from neuronmi.simulators.solver.embedding import EmbeddedMesh
from neuronmi.simulators.solver.membrane import ODESolver
from neuronmi.mesh.mesh_utils import load_h5_mesh
import itertools, unittest, os, subprocess
from neuronmi.mesh.mesh_utils import EMIEntityMap
from dolfin import *
import numpy as np


class TestMembrane(unittest.TestCase):
    mesh_path = './sandbox/test_2neuron.h5'

    @classmethod
    def setUpClass(cls):
        not os.path.exists(TestMembrane.mesh_path) and subprocess.call(['python two_neurons.py'], cwd='./sandbox', shell=True)

    problem_parameters = {'neuron_0': {'I_ion': Constant(0),
                                       'cond': 1,
                                       'C_m': 1,
                                       'stim_strength': 0.0,
                                       'stim_start': 0.0,  
                                       'stim_pos': 0.0,
                                       'stim_length': 0.0},
                          #
                          'neuron_1': {'I_ion': Constant(0),
                                       'cond': 1,
                                       'C_m': 1,
                                       'stim_strength': 0.0,
                                       'stim_start': 0.0,  
                                       'stim_pos': 0.0,
                                       'stim_length': 0.0},
    }
    
    def test_membrane(self):
        emi_map = './sandbox/test_2neuron.json'
        with open(emi_map) as json_fp:
            emi_map = EMIEntityMap(json_fp=json_fp)

        for is_okay in _membrane(TestMembrane.mesh_path, emi_map, TestMembrane.problem_parameters):
            self.assertTrue(is_okay)




set_log_level(WARNING)

def _membrane(mesh_path, emi_map, problem_parameters):
    '''The work horse; running only the ODE part of EMI.'''
    mesh_path = str(mesh_path)
    mesh, volume_marking_f, facet_marking_f = load_h5_mesh(mesh_path)

    num_neurons = emi_map.num_neurons
    # Do we have properties for each one
    neuron_props = [problem_parameters['neuron_%d' % i] for i in range(num_neurons)]

    cell = mesh.ufl_cell()
    Qel = FiniteElement('Discontinuous Lagrange Trace', cell, 0)

    Q = FunctionSpace(mesh, Qel)  
    p0 = Function(Q)              

    n_Stags = map(list,
                  [emi_map.surface_physical_tags('neuron_%d' % i).values() for i in range(num_neurons)])
    all_neuron_surfaces = sum(list(n_Stags), [])

    dt_fem, dt_ode = 1, 1E-2
    fem_ode_sync = 10

    # Mesh of all neurons; individual are its submesh
    neuron_surf_mesh = EmbeddedMesh(facet_marking_f, list(all_neuron_surfaces))
    neurons_subdomains = neuron_surf_mesh.marking_function
    # It is on this mesh that ode will update transmemebrane current and
    # talk with pde
    Q_neuron = FunctionSpace(neuron_surf_mesh, 'DG', 0)  # P0 on surface <-> DLT on facets

    Q = FunctionSpace(mesh, Qel)  # Everywhere
    p0 = Function(Q)              # Previous transm potential now 0

    
    transfer = SubMeshTransfer(mesh, neuron_surf_mesh)
    # The ODE solver talks to the worlk via chain: Q_neuron <-> Q <- W
    p0_neuron = Function(Q_neuron)
    # Between DLT mesh and submesh space
    assign_toQ_neuron_fromQ = transfer.compute_map(Q_neuron, Q, strict=False)
    assign_toQ_fromQ_neuron = transfer.compute_map(Q, Q_neuron, strict=False)
    # From component to DLT on mesh

    toQin_fromQns, toQn_fromQins, p0is = [], [], []
    # p0i \in Qi <-> Q_neuron \ni p0_neuron
    neuron_solutions = []
    for i, neuron_surfaces in enumerate(n_Stags):
        # Pick the nueuron from neuron collection
        ni_mesh = EmbeddedMesh(neurons_subdomains, neuron_surfaces)
        ni_subdomains = ni_mesh.marking_function

        map_ =  emi_map.surface_physical_tags('neuron_%d' % i)

        soma = tuple(map_[k] for k in map_ if 'soma' in k)
        dendrite = tuple(map_[k] for k in map_ if 'dend' in k)
        axon = tuple(map_[k] for k in map_ if 'axon' in k)
        
        ode_solver = ODESolver(ni_subdomains,
                               soma=soma, axon=axon, dendrite=dendrite,
                               problem_parameters=problem_parameters['neuron_%d' % i])

        Tstop = 1.0
        interval = (0.0, Tstop)
    
        # NOTE: a generator; nothing is computed so far
        ode_solutions = ode_solver.solve(interval, dt_ode)  # Potentials only
        neuron_solutions.append(ode_solutions)

        transfer = SubMeshTransfer(neuron_surf_mesh, ni_mesh)
        # Communication between neuron and the collection
        Qi_neuron = ode_solver.V
        p0i_neuron = Function(Qi_neuron)

        # Between DLT mesh and submesh space
        assign_toQin_fromQn = transfer.compute_map(Qi_neuron, Q_neuron, strict=False)
        assign_toQn_fromQin = transfer.compute_map(Q_neuron, Qi_neuron, strict=False)

        toQin_fromQns.append(assign_toQin_fromQn)
        toQn_fromQins.append(assign_toQn_fromQin)
        p0is.append(p0i_neuron)

    f = Expression('x[0]*x[0]+x[1]*x[1]+x[2]*x[2]', degree=2)
        
    neuron_solutions = itertools.izip(*neuron_solutions)

    p1_neuron = p0_neuron.vector().get_local()
    
    step_count = 0
    for odes in neuron_solutions:
        step_count += 1

        (t0, t1) = odes[0][0]

        if step_count == fem_ode_sync:
            step_count = 0
            # From individual neuron to collection
            # Just want not zero
            norms = []
            for i in range(num_neurons):
                norms.append(odes[i][1].vector().norm('l2') > 1E-14)
                # FIXME: does this override?
                toQn_fromQins[i](p0_neuron, odes[i][1])

            neuron_okay = all(norms) and p0_neuron.vector().norm('l2') > 1E-1
            # Upscale p0_neuron->p0
            assign_toQ_fromQ_neuron(p0, p0_neuron)

            # We should get sommething else between steps
            diff_okay = np.linalg.norm(p0_neuron.vector().get_local() - p1_neuron) > 1E-14
            p1_neuron[:] = p0_neuron.vector().get_local()
            
            # File('fpp.pvd') << p0_neuron
            finite_neuron = not np.isnan(p0_neuron.vector().norm('linf'))
            finite = not np.isnan(p0.vector().norm('linf'))

            yield finite_neuron and finite and neuron_okay and diff_okay

            f.t = t1
            # Fake data
            p0.vector()[:] = interpolate(f, Q).vector()

            assign_toQ_neuron_fromQ(p0_neuron, p0)  # To membrane space

            for i in range(num_neurons):
                toQin_fromQns[i](p0is[i], p0_neuron)
                odes[i][1].assign(p0is[i]) 
