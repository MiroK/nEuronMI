from shapes.utils import first, second


def EMI_model(model, box, neuron, probe=None, tol=1E-10):
    '''Define geometry for EMI simulations (fill the model)'''
    # Neuron is contained
    assert all(box.contains(p, tol) for p in neuron.control_points)

    # There is no collision
    if probe is not None:
        check = not any(neuron.contains(p, tol) for p in probe.control_points)
        assert check and not any(probe.contains(p, tol) for p in neuron.control_points)

        # Also probe should cut the box
        assert any(not box.contains(p, tol) for p in probe.control_points)

    # Without the probe we have neuron as one volume and (box-volume)
    # as the other
    box_tag = box.as_gmsh(model)
    # Otherwise we will refer to box as the difference of box and the probe
    if probe is not None:
        probe_tag = probe.as_gmsh(model)
        # Box - probe
        print box_tag, probe_tag
        box_tag, _ = model.occ.cut([(3, box_tag)], [(3, probe_tag)])
        box_tag = second(box_tag[0])  # Pair (3, tag)

    # There is neuron in any case
    neuron_tag, neuron_surfs = neuron.as_gmsh(model)

    # Two volumes are created due to fragmentation
    vols, _ = model.occ.fragment([(3, box_tag)], [(3, neuron_tag)])
    model.occ.synchronize()
    
    # The neuron is the one whose boundaries are neuron surfaces
    surfs0, surfs1 = map(model.getBoundary, vols)
    # A boundary con also contain curves - we ignore those; only keep 2d
    surfs0 = set(s[1] for s in surfs0 if s[0] == 2)
    surfs1 = set(s[1] for s in surfs1 if s[0] == 2)

    neuron_surfs = set(neuron_surfs)
    if surfs0 == neuron_surfs:
        neuron, external = map(second, (vols[0], vols[1]))
        # NOTE: bdry of external is neuron + box bdry
        box_bdry = surfs1 - neuron_surfs  # This can include probe
    else:
        assert surfs1 == neuron_surfs
            
        neuron, external = map(second, (vols[1], vols[0]))
        box_bdry = surfs0 - neuron_surfs

    # Now we can label each volume as Physical Volume
    model.addPhysicalGroup(3, [neuron], 1)     # Neuron is 1
    model.addPhysicalGroup(3, [external], 2)   # External is 2

        # stag = 1
        # # Start marking; each neuron bit gets a different tag

        # keys, box_surfaces = zip(*box.surfaces.items())
        # box_surfaces = np.array(box_surfaces)
        # # Each external bit gets a part
        # for surf in box_bdry:
        #     p = np.array(model.occ.getCenterOfMass(2, surf))
        #     dist = np.linalg.norm(box_surfaces - p, axis=1)
        #     imin = np.argmin(dist)
        #     # A successfull pairing
        #     #if dist[imin] < 1E-13:
                
            
        # surfaces = [(surf, ) for surf in box_bdry]

        # from scipy.spatial import distance_matrix


        #print surfaces
        #M = distance_matrix(map(second, surfaces), box_surfaces)
        #print M
        # match_surfaces(surfaces, box)
        
        # The surfaces as Physical Surface based on whether neuron or
        # the rest claims them
        
        ##print neuron_surfs
        #print model.occ.getCenterOfMass(2, x)
        
        #print model.getBoundary(vols[0])

        #print model.getBoundary(vols[1])

        # TODO: Tagging

    # TODO: With probe, tagging

    # TODO: fields
# --------------------------------------------------------------------

if __name__ == '__main__':
    from shapes import *
    import numpy as np
    import sys


    box = Box(np.array([-3.5, -3, -5]), np.array([6, 6, 10]))
    neuron = TaperedNeuron()
    probe = MicrowireProbe({'tip_x': 1.5, 'radius': 0.2, 'length': 10})
    import gmsh


    model = gmsh.model
    factory = model.occ

    gmsh.initialize(sys.argv)

    gmsh.option.setNumber("General.Terminal", 1)

    EMI_model(model, box, neuron, probe)    
    factory.synchronize();

    # model.mesh.generate(3)
    #model.mesh.refine()
    #model.mesh.setOrder(2)
    #model.mesh.partition(4)
    
    #gmsh.write("neuron.msh")
    gmsh.fltk.initialize()
    gmsh.fltk.run()
    gmsh.finalize()
