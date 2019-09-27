from dolfin import Mesh, MeshFunction, HDF5File, mpi_comm_world
import meshconvert
import numpy as np
import os


def msh_to_h5(msh_file, clean_xml=True):
    '''Convert from msh to h5'''
    root, _ = os.path.splitext(msh_file)
    assert os.path.splitext(msh_file)[1] == '.msh'
    
    # Get the xml mesh
    xml_file = '.'.join([root, 'xml'])

    # Convert to XML
    meshconvert.convert2xml(msh_file, xml_file, iformat='gmsh')

    # Success?
    assert os.path.exists(xml_file)

    mesh = Mesh(xml_file)
    h5_file = '.'.join([root, 'h5'])
    out = HDF5File(mesh.mpi_comm(), h5_file, 'w')
    out.write(mesh, 'mesh')

    # Save ALL data as facet_functions
    data_sets = ('curves', 'surfaces', 'volumes')
    regions = ('curve_region.xml', 'facet_region.xml', 'volume_region.xml')

    for data_set, region in zip(data_sets, regions):
        r_xml_file = '_'.join([root, region])

        if os.path.exists(r_xml_file):
            f = MeshFunction('size_t', mesh, r_xml_file)
            out.write(f, data_set)

            clean_xml and os.remove(r_xml_file)
    clean_xml and os.remove(xml_file)

    return h5_file


def load_h5_mesh(h5_file):
    '''Unpack to mesh, volumes and surfaces'''

    comm = mpi_comm_world()
    h5 = HDF5File(comm, h5_file, 'r')
    mesh = Mesh()
    h5.read(mesh, 'mesh', False)

    surfaces = MeshFunction('size_t', mesh, mesh.topology().dim()-1)
    h5.read(surfaces, 'surfaces')

    volumes = MeshFunction('size_t', mesh, mesh.topology().dim())
    h5.read(volumes, 'volumes')

    return mesh, volumes, surfaces

# --------------------------------------------------------------------

if __name__ == '__main__':
    # Demo of generating mesh from model
    from dolfin import File
    
    root = 'test_neuron'
    msh_file = '%s.msh' % root
    if not os.path.exists(msh_file):
        
        from emi_geometry import build_EMI_geometry, mesh_config_EMI_model
        from shapes import Box, BallStickNeuron, MicrowireProbe
        import gmsh, sys, json, os
        import numpy as np

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
        # 3d model
        model.mesh.generate(3)
        # Native optimization
        model.mesh.optimize('')
        gmsh.write(msh_file)
        gmsh.finalize()

    # Convert
    h5_file = msh_to_h5(msh_file)

    mesh, volumes, surfaces = load_h5_mesh(h5_file)

    assert set(volumes.array()) == set((1, 2))
    assert set(surfaces.array()) == set(range(14))

    # Eye check
    # File('surfaces.pvd') << surfaces
    # File('volumes.pvd') << surfaces

    # Dupliecate vertices?
    from scipy.spatial import distance_matrix
    
    x = mesh.coordinates()
    M = distance_matrix(x, x)

    hmin = mesh.hmin()/2
    # No duplicates
    print all(np.min(row[i+1:]) > hmin for i, row in zip(range(len(x)-2), M))
