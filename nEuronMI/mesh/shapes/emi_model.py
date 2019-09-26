
def EMI_model(model, box, neuron, probe=None, tol=1E-10):
    '''Define geometry for EMI simulations (fill the model)'''
    # Neuron is contained
    assert all(box.contains(p, tol) for p in neuron.control_points)

    # There is no collision
    if probe is not None:
        check = not any(neuron.contains(p, tol) for p in probe.control_points)
        assert check and not any(probe.contains(p, tol) for p in neuron.control_points)


    # Without the probe we have neuron as one volume and (box-volume)
    # as the other
    if probe is None:
        box_tag = box.as_gmsh(model)
        neuron_tag, neuron_surfs = neuron.as_gmsh(model)
        print
        print model.occ.fragment([(3, box_tag)], [(3, neuron_tag)])

        # TODO: Tagging

    # TODO: With probe, tagging

    # TODO: fields
# --------------------------------------------------------------------

if __name__ == '__main__':
    from taperedneuron import TaperedNeuron
    from microwireprobe import MicrowireProbe
    from gmsh_primitives import Box
    import numpy as np
    import sys


    box = Box(np.array([-3.5, -3, -4]), np.array([6, 6, 8]))
    neuron = TaperedNeuron()

    import gmsh


    model = gmsh.model
    factory = model.occ

    gmsh.initialize(sys.argv)
    gmsh.fltk.initialize()
    gmsh.option.setNumber("General.Terminal", 1)

    EMI_model(model, box, neuron)    
    factory.synchronize();

    #model.mesh.generate(3)
    #model.mesh.refine()
    #model.mesh.setOrder(2)
    #model.mesh.partition(4)
    
    #gmsh.write("neuron.msh")
    gmsh.fltk.run()
    gmsh.finalize()
