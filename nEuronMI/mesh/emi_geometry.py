from shapes.utils import first, second
from shapes.baseneuron import Neuron
from itertools import count, chain
from collections import deque
from dolfin import info
from emi_map import EMIEntityMap


def build_EMI_geometry(model, box, neurons, probe=None, tol=1E-10):
    '''
    Define geometry for EMI simulations (fill the model). Return model 
    and mapping of external surfaces [EMIEntityMap]
    '''
    if isinstance(neurons, Neuron): neurons = [neurons]

    # Neurons are contained
    for neuron in neurons: 
        assert all(box.contains(p, tol) for p in neuron.control_points)

    # There is no collision
    if probe is not None:
        # Between probe and neurons
        for neuron in neurons:
            check = not any(neuron.contains(p, tol) for p in probe.control_points)
            assert check and not any(probe.contains(p, tol) for p in neuron.control_points)

        # And neurons themselves
        for i, n0 in enumerate(neurons):
            for n1 in neurons[i+1:]:
                assert not any(n1.contains(p, tol) for p in n0.control_points)

        # Also probe should cut the box
        assert any(not box.contains(p, tol) for p in probe.control_points)

    # Without the probe we have neurons as volumes and (box-volume) as the other
    box_tag = box.as_gmsh(model)
    # Otherwise we will refer to box as the difference of box and the probe
    if probe is not None:
        probe_tag = probe.as_gmsh(model)
        # Box - probe; cur returns volumes and sufs. volume is pair (3, tag)
        [(_, box_tag)] = first(model.occ.cut([(3, box_tag)], [(3, probe_tag)]))

    # There is neuron in any case; so we add them to medel as volumes with
    # neuron_tags. The surfaces are only auxiliary as they will change during cut
    neurons_tags, neurons_surfs = zip(*(neuron.as_gmsh(model) for neuron in neurons))
    # The volumes are created due to fragmentation
    volumes = first(model.occ.fragment([(3, box_tag)], [(3, nt) for nt in neurons_tags]))
    model.occ.synchronize()

    # Now we would like to find in volumes the neurons and extecellular domain
    # The idea is that a neuron is a closed volume whose boundary has neuron surfaces
    volumes_surfs = map(model.getBoundary, volumes)
    # A boundary can also contain curves - we ignore those; only keep 2d
    volumes_surfs = [set(s[1] for s in ss if s[0] == 2) for ss in volumes_surfs]

    volume_pairs = zip(volumes, volumes_surfs)
    # Volume with largest bdry is external
    external_pair = max(volume_pairs, key=lambda vs: len(second(vs)))
    external_volume, external_surfs = external_pair

    neuron_mapping = []  # i-th neuron is neuron_mapping[i] volume and has surfaces ...
    # Ther rest are neurons
    volume_pairs.remove(external_pair)
    volume_pairs = deque(volume_pairs)
    for i, neuron in enumerate(neurons):
        match = False
        neuron_surfaces = {}  # Find them in the bounding surfaces of model

        while not match:
            pair = volume_pairs.pop()
            vol_tags, vol_surfs = pair
            # Eliminite geom. checks if the number of surfs doesn' t match
            if len(vol_surfs) != len(neurons_surfs[i]):
                volume_pairs.appendleft(pair)
                continue
            # Try geom check
            match = neuron.link_surfaces(model, vol_surfs, links=neuron_surfaces, tol=tol)
            # If some found then all found
            assert not match or set(neuron_surfaces.keys()) == set(neuron.surfaces.keys()), (neuron, neuron_surfaces)
            # If success we can pair say that i-th neuron is that volume
            if match:
                neuron_mapping.append((vol_tags, neuron_surfaces))
                # We can remove the surfaces froma all bounding
                external_surfs.difference_update(neuron_surfaces.values())
            # Try with a different neuron
            else:
                volume_pairs.appendleft(pair)
                
    # At this point the external_surfs either belong to the box or the probe
    probe_surfaces = {}
    if probe is not None:
        probe.link_surfaces(model, external_surfs, box=box, tol=tol, links=probe_surfaces)
        assert set(probe_surfaces.keys()) == set(probe.surfaces.keys())
    
    box_surfaces = {}
    box.link_surfaces(model, external_surfs, links=box_surfaces)
    # Success, what box wanted was found
    assert set(box_surfaces.keys()) == set(box.surfaces.keys())

    external_surfs and info('There are unclaimed surfaces % s' % external_surfs)

    # Finally we assign physical groups: name -> {name -> (entity tag, physical tag)}
    vtags, stags = count(1), count(1)
    # Dictionary with geom tags for entities -> (geom, phys. entities)
    tag_surfaces = lambda d_entity, tags=stags: {k: (d_entity[k], tag) for k, tag in zip(d_entity, tags)}
    
    tagged_volumes = {'external': {'all': (second(external_volume), next(vtags))}}
    tagged_surfaces = {'box': tag_surfaces(box_surfaces), 'probe': tag_surfaces(probe_surfaces)}

    for i, (neuron_volume, neuron_surfaces) in enumerate(neuron_mapping):
        tagged_volumes['neuron_%d' % i] = {'all': (second(neuron_volume), next(vtags))}
        tagged_surfaces['neuron_%d' % i] = tag_surfaces(neuron_surfaces)

    # Add to model
    for (etag, ptag) in chain(*[d.values() for d in tagged_volumes.values()]):
        model.addPhysicalGroup(3, [etag], ptag)

    for (etag, ptag) in chain(*[d.values() for d in tagged_surfaces.values()]):
        model.addPhysicalGroup(2, [etag], ptag)

    return model, EMIEntityMap(tagged_volumes, tagged_surfaces)


def mesh_config_EMI_model(model, mapping, mesh_sizes):
    '''Extend model by adding mesh size info'''
    # NOTE: this is really now just an illustration of how it could be
    # done. With the API the model can be configured in any way
    
    field = model.mesh.field
    # We want to specify size for neuron surfaces and surfaces of the
    # probe separately; let's collect them
    neuron_surfaces = mapping.surface_entity_tags('all_neurons').values()

    field.add('MathEval', 1)
    field.setString(1, 'F', str(mesh_sizes['neuron']))
    field.add('Restrict', 2)
    field.setNumber(2, 'IField', 1)
    field.setNumbers(2, 'FacesList', neuron_surfaces)

    next_field = 2
    probe_surfaces = mapping.surface_entity_tags('probe').values()
    if probe_surfaces:
        next_field += 1
        field.add('MathEval', next_field)
        field.setString(next_field, 'F', str(mesh_sizes['probe']))
        
        next_field += 1
        field.add('Restrict', next_field)
        field.setNumber(next_field, 'IField', next_field-1)
        field.setNumbers(next_field, 'FacesList', probe_surfaces)

    box_surfaces = mapping.surface_entity_tags('box').values()
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
    box = Box(np.array([-3.5, -3, -5]), np.array([8, 8, 13]))
    neurons = [TaperedNeuron({'dend_rad': 0.2, 'axon_rad': 0.3}),
               BallStickNeuron({'soma_x': -2, 'soma_y': -2, 'soma_z': 2}),
               BallStickNeuron({'soma_x': 2, 'soma_y': 2, 'soma_z': 2})]

    probe = MicrowireProbe({'tip_x': 1.5, 'radius': 0.2, 'length': 10})

    mesh_sizes = {'neuron': 0.3, 'probe': 0.1, 'box': 1}
    
    model = gmsh.model
    factory = model.occ
    # You can pass -clscale 0.25 (to do global refinement)
    # or -format msh2            (to control output format of gmsh)
    gmsh.initialize(sys.argv)

    gmsh.option.setNumber("General.Terminal", 1)

    # Add components to model
    model, mapping = build_EMI_geometry(model, box, neurons, probe)
    # Dump the mapping as json
    with open('%s.json' % root, 'w') as out:
        mapping.dump(out)

    with open('%s.json' % root) as json_fp:
        mapping = EMIEntityMap(json_fp=json_fp)

    # Add fields controlling mesh size
    mesh_config_EMI_model(model, mapping, mesh_sizes)
    
    factory.synchronize();
    # This is a way to store the geometry as geo file
    gmsh.write('%s.geo_unrolled' % root)
    # Launch gui for visual inspection
    gmsh.fltk.initialize()
    gmsh.fltk.run()

    gmsh.finalize()
