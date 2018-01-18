import cbcbeat as beat
from dolfin import *


# FIXME: Karoline can you get this in shape?
def ODESolver(subdomains, soma, axon, dendrite, parameters):
    '''
    Setup a membrane model and an ODE solver for it.
    '''
    # This is just a mock up but MultiCellModel seems to be the right
    # ingredient
    soma_model = beat.Beeler_reuter_1977()
    axon_model = beat.Beeler_reuter_1977()
    dendrite_model = beat.Beeler_reuter_1977()

    
    neuron_model = beat.MultiCellModel((soma_model, axon_model, dendrite_model),
                                       (soma, axon, dendrite),
                                       subdomains)

    mesh = subdomains.mesh()
    time = Constant(0.0)  # Start from now
    ode_solver = beat.BasicCardiacODESolver(mesh, time, neuron_model,
                                            I_s=Expression('exp(-t)', t=time, degree=1))

    return ode_solver, neuron_model
