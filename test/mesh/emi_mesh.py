from dolfin import Mesh, MeshFunction, HDF5File, MPI

    root = 'test_neuron'
    msh_file = '%s.msh' % root
        
    # Components
    # box = Box(np.array([-100, -100, -100]), np.array([200, 200, 400]))
    
    # neurons = [BallStickNeuron({'soma_x': 20, 'soma_y': 20, 'soma_z': 0,
    #                             'soma_rad': 20, 'dend_len': 50, 'axon_len': 50,
    #                             'dend_rad': 15, 'axon_rad': 10}),
    #            TaperedNeuron({'soma_x': 30, 'soma_y': -30, 'soma_z': 20,
    #                           'soma_rad': 20, 'dend_len': 20, 'axon_len': 20, 'axonh_len': 30, 'dendh_len': 20,
    #                           'dend_rad': 10, 'axon_rad': 8, 'axonh_rad': 10, 'dendh_rad': 15})]
    
    # # probe = MicrowireProbe({'tip_x': -20, 'radius': 5, 'length': 400})
    # probe = Neuropixels24Probe({'tip_x': 80, 'length': 1200, 'angle': 0})
    # # Coarse enough for tests
    # size_params = {'DistMax': 20, 'DistMin': 10, 'LcMax': 10,
    #                'neuron_LcMin': 4, 'probe_LcMin': 2}

    # This gives course enough mesh that the solver runs fast
    box = Box(np.array([-100, -100, -100]), np.array([200, 200, 200]))
    
    neurons = [BallStickNeuron({'soma_x': 0, 'soma_y': 0, 'soma_z': 0,
                                'soma_rad': 20, 'dend_len': 50, 'axon_len': 50,
                                'dend_rad': 15, 'axon_rad': 10}),
               TaperedNeuron({'soma_x': 30, 'soma_y': -30, 'soma_z': 20,
                              'soma_rad': 20, 'dend_len': 20, 'axon_len': 20, 'axonh_len': 30, 'dendh_len': 20,
                              'dend_rad': 10, 'axon_rad': 8, 'axonh_rad': 10, 'dendh_rad': 15})]
    
    probe = MicrowireProbe({'tip_x': -20, 'radius': 5, 'length': 400})
    probe = NeuronexusProbe({'tip_x': 50, 'length': 1200, 'angle': 0})

    # Coarse enough for tests
    size_params = {'DistMax': 20, 'DistMin': 10, 'LcMax': 40,
                   'neuron_LcMin': 20, 'probe_LcMin': 10}
    
    model = gmsh.model
    factory = model.occ
    # You can pass -clscale 0.25 (to do global refinement)
    # or -format msh2            (to control output format of gmsh)
    args = sys.argv + ['-format', 'msh2']  # Dolfin convert handles only this
    gmsh.initialize(args)

    gmsh.option.setNumber("General.Terminal", 1)

    # Add components to model
    model, mapping = build_EMI_geometry(model, box, neurons, probe=probe)

    with open('%s.json' % root, 'w') as out:
        mapping.dump(out)

    # Dump the mapping as json
    mesh_config_EMI_model(model, mapping, size_params)
    factory.synchronize()

    # gmsh.fltk.initialize()
    # gmsh.fltk.run()

    # This is a way to store the geometry as geo file
    gmsh.write('%s.geo_unrolled' % root)
    # 3d model
    model.mesh.generate(3)
    # Native optimization
    model.mesh.optimize('')
    gmsh.write(msh_file)
    gmsh.finalize()
    
    # Convert
    h5_file = msh_to_h5(msh_file)

    mesh, volumes, surfaces = load_h5_mesh(h5_file)

    # Eye check
    File('surfaces.pvd') << surfaces
    File('volumes.pvd') << volumes
