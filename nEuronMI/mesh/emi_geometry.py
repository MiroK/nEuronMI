from shapes.utils import first, second
from itertools import count, chain


def build_EMI_geometry(model, box, neuron, probe=None, tol=1E-10):
    '''
    Define geometry for EMI simulations (fill the model). Return model 
    and mapping of external surfaces (to be used for bcs etc)
    '''
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
        neuron_tag, external_tag = map(second, (vols[0], vols[1]))
        # NOTE: bdry of external is neuron + box bdry
        box_bdry = surfs1 - neuron_surfs  # This can include probe
    else:
        assert surfs1 == neuron_surfs
            
        neuron_tag, external_tag = map(second, (vols[1], vols[0]))
        box_bdry = surfs0 - neuron_surfs

    # Let neuron, probe and box claim the external surfaces
    neuron_surfaces = {}
    neuron.link_surfaces(model, neuron_surfs, links=neuron_surfaces, tol=tol)
    # Success, what neuron wanted was found
    assert not neuron_surfs
    assert set(neuron_surfaces.keys()) == set(neuron.surfaces.keys())

    probe_surfaces = {}
    if probe is not None:
        probe.link_surfaces(model, box_bdry, box=box, tol=tol, links=probe_surfaces)
        assert set(probe_surfaces.keys()) == set(probe.surfaces.keys())
    
    # And the box
    external_surfaces = {}
    box.link_surfaces(model, box_bdry, links=external_surfaces)
    # Success, what box wanted was found
    assert set(external_surfaces.keys()) == set(box.surfaces.keys())

    # Finally we assign physical groups
    # Volumes
    tagged_entities = {3: {'neuron': {'all': (neuron_tag, 1)},
                           'external': {'all': (external_tag, 2)}}}
    # Now sequentially surfaces
    shapes = {'neuron': neuron_surfaces, 'box': external_surfaces, 'probe': probe_surfaces}
    
    tags = count(1)
    tagged_surfaces = {shape: {k: (shape_surfs[k], tag) for k, tag in zip(shape_surfs, tags)}
                       for shape, shape_surfs in shapes.items()}

    tagged_entities[2] = tagged_surfaces

    # Update the model
    for dim, entities in tagged_entities.items():
        # Don't care about grouping here
        for (etag, ptag) in chain(*[d.values() for d in entities.values()]):
            model.addPhysicalGroup(dim, [etag], ptag)
    
    return model, tagged_entities


def mesh_config_EMI_model(model, mapping, mesh_sizes):
    '''Extend model by adding mesh size info'''
    # NOTE: this is really now just an illustration of how it could be
    # done. With the API the model can be configured in any way
    
    field = model.mesh.field
    # We want to specify size for neuron surfaces and surfaces of the
    # probe separately; let's collect them
    neuron_surfaces = map(second, mapping[2]['neuron'].values())

    field.add('MathEval', 1)
    field.setString(1, 'F', str(mesh_sizes['neuron']))
    field.add('Restrict', 2)
    field.setNumber(2, 'IField', 1)
    field.setNumbers(2, 'FacesList', neuron_surfaces)

    next_field = 2
    probe_surfaces = mapping[2]['probe'].values()
    if probe_surfaces:
        probe_surfaces = map(second, probe_surfaces)
        
        next_field += 1
        field.add('MathEval', next_field)
        field.setString(next_field, 'F', str(mesh_sizes['probe']))
        
        next_field += 1
        field.add('Restrict', next_field)
        field.setNumber(next_field, 'IField', next_field-1)
        field.setNumbers(next_field, 'FacesList', probe_surfaces)

    box_surfaces = map(second, mapping[2]['box'].values())
    next_field += 1
    field.add('MathEval', next_field)
    field.setString(next_field, 'F', str(mesh_sizes['box']))
        
    next_field += 1
    field.add('Restrict', next_field)
    field.setNumber(next_field, 'IField', next_field-1)
    field.setNumbers(next_field, 'FacesList', box_surfaces)

    next_field += 1
    fields = list(range(2, next_field, 2))

    # Where they meet
    field.add('Min', next_field)
    field.setNumbers(next_field, 'FieldsList', fields)
    field.setAsBackgroundMesh(next_field)

    return model

# --------------------------------------------------------------------

if __name__ == '__main__':
    # Demo of working with Gmsh
    from shapes import *
    import numpy as np
    import gmsh, sys, json

    root = 'test_neuron'
    # Components
    box = Box(np.array([-3.5, -3, -5]), np.array([6, 6, 10]))
    neuron = BallStickNeuron()
    probe = MicrowireProbe({'tip_x': 1.5, 'radius': 0.2, 'length': 10})

    mesh_sizes = {'neuron': 0.3, 'probe': 0.1, 'box': 1}
    
    model = gmsh.model
    factory = model.occ
    # You can pass -clscale 0.25 (to do global refinement)
    # or -format msh2            (to control output format of gmsh)
    gmsh.initialize(sys.argv)

    gmsh.option.setNumber("General.Terminal", 1)

    # Add components to model
    model, mapping = build_EMI_geometry(model, box, neuron, probe)
    # Dump the mapping as json
    with open('%s.json' % root, 'w') as out:
        json.dump(mapping, out)
    # Add fields controlling mesh size
    mesh_config_EMI_model(model, mapping, mesh_sizes)
    
    factory.synchronize();
    # This is a way to store the geometry as geo file
    gmsh.write('%s.geo_unrolled' % root)
    # Launch gui for visual inspection
    gmsh.fltk.initialize()
    gmsh.fltk.run()

    gmsh.finalize()
