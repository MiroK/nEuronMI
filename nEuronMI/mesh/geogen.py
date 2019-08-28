import hashlib, time, os
from os.path import join
import subprocess, os
import numpy as np
from .shapes import SphereNeuron, MainenNeuron # rename
from .shapes import CylinderProbe, BoxProbe, WedgeProbe, FancyProbe, PixelProbe # rename
from math import pi
import sys

GEO_CODE_DIR = os.path.join(os.path.dirname(__file__), 'geo_codes')

def header_code(neuron, sizes, probe=None):
    '''Write parameter definitions for Gmsh'''
    if probe is None:
        assert all(k in sizes for k in ('neuron_mesh_size', 'rest_mesh_size'))
        all_params = {}
    else:
        assert all(k in sizes for k in ('neuron_mesh_size', 'rest_mesh_size', 'probe_mesh_size'))
        all_params = dict(probe.params)

    all_params.update(neuron.params)
    all_params.update(sizes)

    code = ['DefineConstant[']
    code += ['%s = {%g, Name "%s"}' % (k, v, k) for k, v in all_params.items()]
    code += ['];']
    code.append('SetFactory("OpenCASCADE");')

    return '\n'.join(code)


def read_code(dispatch, directory=GEO_CODE_DIR):
    '''Geometries for gmsh are taken from files'''
    with open(os.path.join(directory, dispatch)) as f:
        code = f.readlines()
    code = '\n'.join(code)
    
    return code


def mesh_size_code(has_probe):
    '''It is assumed that all the surfaces have been defined in the code'''
    code = '''
Field[1] = MathEval;
Field[1].F = Sprintf("%g", neuron_mesh_size);

Field[2] = Restrict;
Field[2].IField = 1;
Field[2].FacesList = {neuron_surface[]};
'''

    if has_probe:
        code += '''
// Mesh size on Probe
Field[3] = MathEval;
Field[3].F = Sprintf("%g", probe_mesh_size);

Field[4] = Restrict;
Field[4].IField = 3;
Field[4].FacesList = {probe_surface[]};
  
// Mesh size everywhere else
Field[5] = MathEval;
Field[5].F = Sprintf("%g", rest_mesh_size);
  
Field[6] = Min;
Field[6].FieldsList = {2, 4, 5};
Background Field = 6;  
'''
    else:
        code += '''
// Mesh size everywhere else
Field[3] = MathEval;
Field[3].F = Sprintf("%g", rest_mesh_size);

Field[4] = Min;
Field[4].FieldsList = {2, 3};
Background Field = 4;
'''
    return code

# ---

def geofile(neuron, sizes, file_name='', probe=None, hide_neuron=False):
    '''
    Write the geo file for given neuron [probe] and mesh sizes. Note that
    all the code pieces rely on existence of variables like rad_soma etc ...
    '''
    # Probe must not cross neuron 
    assert probe is None or not any(neuron.is_inside(p) for p in probe.control_points())
    # or leave the bbox. Even if passes, the
    assert probe is None or all(neuron.is_inside_bbox(p) for p in probe.control_points())
    # NOTE: probe might be too close to the bbox/neuron
    
    # Code gen
    # Definition of all the fields
    header = header_code(neuron, sizes, probe)

    # Special defs of the neuron which may not be user params
    neuron_defs = neuron.definitions(hide_neuron)
    
    # The neuron - relies on defined vars
    neuron_code = read_code(str(neuron))
    # Code when probe included
    if probe is not None:
        # Special defs of the neuron which may not be user params
        probe_defs = probe.definitions(neuron)

        neuron_probe_code = read_code('_'.join(map(str, (neuron, probe))))

        size_code = mesh_size_code(True)
    else:
        probe_defs = ''
        
        neuron_probe_code = '''
outside() = BooleanDifference { Volume{bbox}; Delete; }{ Volume{neuron};};
Physical Volume(2) = {outside[]};  
'''
        size_code = mesh_size_code(False)

    delim = '\n' + '//----' + '\n'
    code = '//\n'.join([header,
                        neuron_defs, neuron_code,
                        probe_defs, neuron_probe_code, size_code])
    
    # File
    if not file_name:
        geo_file = hashlib.sha1()
        geo_file.update(str(time.time()))
        geo_file = geo_file.hexdigest()
    else:
        geo_file = file_name
    geo_file = '.'.join([geo_file, 'GEO'])

    print('Generated geo file', geo_file)
    print('Inspect with Gmsh or run e.g. `gmsh -3 $GEOFILE` to generate mesh')

    with open(geo_file, 'w') as f: f.write(code)

    return geo_file

# -------------------------------------------------------------------

def generate_mesh(neuron_type='tapered', probe_type='neuronexus', microwire_radius=30, distance=50, probe_tip=None,
                  coarseness=2, box_size=2, neuron_params=None, save_mesh_path=None):
    '''
    Parameters
    ----------
    neuron_type: str or None
        The neuron type ([bas' (ball-ans-stick) or 'tapered' (tapered dendrite and axon)]
        If None, a mesh without neuron is generated.
    probe_type: str or None
        The probe type ('microwire', 'neuronexus', 'neuropixels-24')
        If None, a mesh without probe is generated.
    microwire_radius: float
        If probe is 'microwire', the microwire radius
    distance: float
        Distance in um between the center of the neuron and the probe in the x-direction
    probe_tip: list or np.array
        The 3d position of the probe tip
    coarseness: int or dict
        Coarseness of the mesh. It can be 00, 0, 1, 2, 3 (less course to more coarse) or
        a dictionary with 'neuron', 'probe', 'rest' fields with cell size in um
    box_size: int or dict
        Size of the boundig box. It can be 1, 2, 3, 4, 5, 6 (smaller to larger) or
        a dictionary with 'dx', 'dy', 'dz' (scalar or vector of 2), which are the distances from the
        neuron end in each direction to the bounding box
    neuron_params: dict
        Dictionary with neuron params: 'rad_soma', 'rad_dend', 'rad_axon', 'len_dend', 'len_axon'.
        If the 'neuron_type' is 'tapered', also 'rad_dend_base' and 'rad_axon_base'
    save_mesh_path: str or Path
        The output path. If None, a 'mesh' folder is created in the current working directory.
    Returns
    -------
    mesh_h5: str
        Path to the h5-converted mesh, ready for simulation
    '''
    # generate
    if isinstance(box_size, int):
        dx, dy, dz = return_boxsizes(box_size)
    else:
        dx = box_size['dx']
        dy = box_size['dx']
        dz = box_size['dx']
    
    if isinstance(coarseness, int):
        nmesh, pmesh, rmesh = return_coarseness(coarseness)
    else:
        nmesh = coarseness['neuron']
        pmesh = coarseness['probe']
        rmesh = coarseness['rest']
    
    if neuron_params is None:
        geometrical_params = {'rad_soma': 10 * conv, 'rad_dend': 2.5 * conv, 'rad_axon': 1 * conv,
                              'length_dend': 400 * conv, 'length_axon': 200 * conv, 'rad_hilox_d': 4 * conv,
                              'length_hilox_d': 20 * conv, 'rad_hilox_a': 2 * conv, 'length_hilox_a': 10 * conv,
                              'dxp': dx * conv, 'dxn': dx * conv, 'dy': dy * conv, 'dz': dz * conv}
    else:
        geometrical_params = neuron_params
        
    mesh_sizes = {'neuron_mesh_size': nmesh * geometrical_params['rad_axon'],
                  'probe_mesh_size': pmesh * geometrical_params['rad_axon'],
                  'rest_mesh_size': rmesh * geometrical_params['rad_axon']}
    
    if neuron_type == 'sphere':
        neuron = SphereNeuron(geometrical_params)
    elif neuron_type == 'mainen':
        neuron = MainenNeuron(geometrical_params)

    probe_x = probe_tip[0] * conv
    probe_y = probe_tip[1] * conv
    probe_z = probe_tip[2] * conv

    print('Probe tip: ', probe_x, probe_y, probe_z)

    if probetype == 'cylinder':
        probe = CylinderProbe({'rad_probe': rad * conv, 'probe_x': probe_x, 'probe_y': probe_y, 'probe_z': probe_z})
    elif probetype == 'fancy':
        # Angle in radians
        probe = FancyProbe({'probe_x': probe_x, 'probe_y': probe_y, 'probe_z': probe_z,
                            'with_contacts': 1, 'rot_angle': np.deg2rad(rot)})
    elif probetype == 'pixel':
        # Angle in radians
        probe = PixelProbe({'probe_x': probe_x, 'probe_y': probe_y, 'probe_z': probe_z,
                            'with_contacts': 1, 'rot_angle': np.deg2rad(rot)})

    if not os.path.isdir(join(root, probetype)):
        os.mkdir(join(root, probetype))
    if not os.path.isdir(join(root, probetype, mesh_name)):
        os.mkdir(join(root, probetype, mesh_name))

    if not hide_neuron:
        fname_wprobe = join(root, probetype, mesh_name, mesh_name + '_wprobe')
        out_wprobe = geofile(neuron, mesh_sizes, probe=probe, file_name=fname_wprobe, hide_neuron=hide_neuron)
        fname_noprobe = join(root, probetype, mesh_name, mesh_name + '_noprobe')
        out_noprobe = geofile(neuron, mesh_sizes, probe=None, file_name=fname_noprobe, hide_neuron=hide_neuron)
    else:
        print('Mesh name', mesh_name)
        if not os.path.isdir(join(root, 'noneuron', probetype, mesh_name)):
            os.makedirs(join(root, 'noneuron', probetype, mesh_name))
            print('Created ', join(root, 'noneuron', probetype, mesh_name))
        fname_wprobe = join(root, 'noneuron', probetype, mesh_name, mesh_name + '_wprobe')
        out_wprobe = geofile(neuron, mesh_sizes, probe=probe, file_name=fname_wprobe, hide_neuron=hide_neuron)

    sys.exit
    import subprocess
    if show:
        subprocess.call(['gmsh %s' % out_wprobe], shell=True)
        if not hide_neuron:
            subprocess.call(['gmsh %s' % out_noprobe], shell=True)
    else:
        subprocess.call(['gmsh -3 %s' % out_wprobe], shell=True)
        if not hide_neuron:
            subprocess.call(['gmsh -3 %s' % out_noprobe], shell=True)

    out_msh_wprobe = out_wprobe[:-3] + 'msh'
    if not hide_neuron:
        out_msh_noprobe = out_noprobe[:-3] + 'msh'

    if return_f:
        sys.exit(out_msh_noprobe + ' ' + out_msh_wprobe)


def return_coarseness(coarse):
    if coarse == 00:
        nmesh = 2
        pmesh = 3
        rmesh = 5
    elif coarse == 0:
        nmesh = 2
        pmesh = 5
        rmesh = 7.5
    elif coarse == 1:
        nmesh = 3
        pmesh = 6
        rmesh = 9
    elif coarse == 2:
        nmesh = 4
        pmesh = 8
        rmesh = 12
    elif coarse == 3:
        nmesh = 4
        pmesh = 10
        rmesh = 15
    else:
        raise Exception('coarseness must be 00, 0, 1, 2, or 3')

    return nmesh, pmesh, rmesh


def return_boxsizes(box):
    if box == 1:
        dx = 80
        dy = 80
        dz = 20
    elif box == 2:
        dx = 100
        dy = 100
        dz = 40
    elif box == 3:
        dx = 120
        dy = 120
        dz = 60
    elif box == 4:
        dx = 160
        dy = 160
        dz = 100
    elif box == 5:
        dx = 200
        dy = 200
        dz = 150
    elif box==6:
        dx = 300
        dy = 300
        dz = 300
    else:
        raise Exception('boxsize must be 1, 2, 3, 4, 5, or 6')

    return dx, dy, dz,


def convert_msh2h5(msh_file, h5_file):
    '''Temporary version of convertin from msh to h5'''
    root, _ = os.path.splitext(msh_file)
    assert os.path.splitext(msh_file)[1] == '.msh'
    assert os.path.splitext(h5_file)[1] == '.h5'

    # Get the xml mesh
    xml_file = '.'.join([root, 'xml'])
    subprocess.call(['dolfin-convert %s %s' % (msh_file, xml_file)], shell=True)
    # Success?
    assert os.path.exists(xml_file)

    cmd = '''from dolfin import Mesh, HDF5File;\
             mesh=Mesh('%(xml_file)s');\
             assert mesh.topology().dim() == 3;\
             out=HDF5File(mesh.mpi_comm(), '%(h5_file)s', 'w');\
             out.write(mesh, 'mesh');''' % {'xml_file': xml_file,
                                            'h5_file': h5_file}

    for region in ('facet_region.xml', 'physical_region.xml'):
        name, _ = region.split('_')
        r_xml_file = '_'.join([root, region])
        if os.path.exists(r_xml_file):
            cmd_r = '''from dolfin import MeshFunction;\
                       f = MeshFunction('size_t', mesh, '%(r_xml_file)s');\
                       out.write(f, '%(name)s');\
                       ''' % {'r_xml_file': r_xml_file, 'name': name}

            cmd = ''.join([cmd, cmd_r])

    cmd = 'python -c "%s"' % cmd

    status = subprocess.call([cmd], shell=True)
    assert status == 0
    # Sucess?
    assert os.path.exists(h5_file)

    return True


def cleanup(files=None, exts=()):
    '''Get rid of xml'''
    if files is not None:
        return map(os.remove, files)
    else:
        files = filter(lambda f: any(map(f.endswith, exts)), os.listdir('.'))
        print('Removing', files)
        return cleanup(files)

if __name__ == '__main__':

    if '-simple' in sys.argv:
        simple=True
    else:
        simple=False
    if '-show' in sys.argv:
        show=True
    else:
        show=False
    if '-returnfname' in sys.argv:
        return_f=True
    else:
        return_f=False
    if '-probetype' in sys.argv:
        pos = sys.argv.index('-probetype')
        probetype = sys.argv[pos + 1]
    else:
        probetype='fancy'
    if '-noneuron' in sys.argv:
        hide_neuron=True
    else:
        hide_neuron=False
    if '-neurontype' in sys.argv:
        pos = sys.argv.index('-neurontype')
        neurontype = sys.argv[pos + 1]
    else:
        neurontype='mainen'
    if '-dist' in sys.argv:
        pos = sys.argv.index('-dist')
        dist = float(sys.argv[pos + 1])
    else:
        dist=40
    if '-rad' in sys.argv:
        pos = sys.argv.index('-rad')
        rad = float(sys.argv[pos + 1])
    else:
	if probetype == 'cylinder':
            rad = 15
	else:
	    rad = 0
    if '-probetip' in sys.argv:
        pos = sys.argv.index('-probetip')
        probetip = sys.argv[pos + 1].split(',')
        probetip = [float(p) for p in probetip]
        print('Probetip: ', probetip)
    else:
        if probetype == 'cylinder':
            probetip=[dist, 0, 0]
        elif probetype == 'fancy':
            probetip=[dist, 0, -100]
        elif probetype == 'pixel':
            probetip=[dist, 0, -200]
    if '-coarse' in sys.argv:
        pos = sys.argv.index('-coarse')
        coarse = int(sys.argv[pos + 1])
    else:
        coarse = 2
    if '-boxsize' in sys.argv:
        pos = sys.argv.index('-boxsize')
        box = int(sys.argv[pos + 1])
    else:
        box = 2
    if '-rot' in sys.argv:
        pos = sys.argv.index('-rot')
        rot = int(sys.argv[pos + 1])
    else:
        rot = 0

    if len(sys.argv) == 1:
        print('Generate GEO and msh files with and without probe. '\
              '\n   -simple : simpler mesh (coarser cells - larger neuron)' \
              '\n   -probetype : cylinder (default) - box - wedge - fancy\n   -neurontype : sphere (default) - mainen' \
              '\n   -probetip : x,y,z of probe tip (in um)\n   -coarse : 1 (less) - 2 - 3 (more)' \
              '\n   -boxsize : 1 (smaller) - 2 - 3 (larger)\n   -rot rotation along z in deg')

        raise Exception('Indicate mesh argumets')
    
    conv=1E-4
    
    if coarse == 00:
        nmesh = 2
        pmesh = 3
        rmesh = 5
    elif coarse == 0:
        nmesh = 2
        pmesh = 5
        rmesh = 7.5
    elif coarse == 1:
        nmesh = 3
        pmesh = 6
        rmesh = 9
    elif coarse == 2:
        nmesh = 4
        pmesh = 8
        rmesh = 12
    elif coarse == 3:
        nmesh = 4
        pmesh = 10
        rmesh = 15
#    if hide_neuron:
#        nmesh = 2
#        pmesh = 5
#        rmesh = 7.5
#        coarse = -2

    if box == 1:
        dxp = 80
        dxn = 80
        dy = 80
        dz = 20
    elif box == 2:
        dxp = 100
        dxn = 100
        dy = 100
        dz = 40
    elif box == 3:
        dxp = 120
        dxn = 120
        dy = 120
        dz = 60
    elif box == 4:
        dxp = 160
        dxn = 160
        dy = 160
        dz = 100
    elif box == 5:
        dxp = 200
        dxn = 200
        dy = 200
        dz = 150
    elif box==6:
        dxp = 300
        dxn = 300
        dy = 300
        dz = 300

    root = os.getcwd()

    # ########################
    # # SIMPLE
    # ########################
    # if simple:
    #     geometrical_params = {'rad_soma': 30 * conv, 'rad_dend': 10 * conv, 'rad_axon': 6 * conv,
    #                           'length_dend': 400 * conv, 'length_axon': 200 * conv, 'rad_hilox_d': 16 * conv,
    #                           'length_hilox_d': 10 * conv, 'rad_hilox_a': 10 * conv, 'length_hilox_a': 10 * conv,
    #                           'dxp': dxp * conv, 'dxn': dxn * conv, 'dy': dy * conv, 'dz': dz * conv}
    #     mesh_sizes = {'neuron_mesh_size': 2 * geometrical_params['rad_axon'],
    #                   'probe_mesh_size': 2 * geometrical_params['rad_axon'],
    #                   'rest_mesh_size': 4 * geometrical_params['rad_axon']}
    # #####################################
    # # DETAILED
    # #####################################
    # else:

    if hide_neuron:
        if probetype == 'cylinder':
            probetip = [0, 0, 0]
        elif probetype == 'fancy':
            probetip = [0, 0, -100]
        elif probetype == 'pixel':
            probetip = [0, 0, -200]

    geometrical_params = {'rad_soma': 10 * conv, 'rad_dend': 2.5 * conv, 'rad_axon': 1 * conv,
                          'length_dend': 400 * conv, 'length_axon': 200 * conv, 'rad_hilox_d': 4 * conv,
                          'length_hilox_d': 20 * conv, 'rad_hilox_a': 2 * conv, 'length_hilox_a': 10 * conv,
                          'dxp': dxp * conv, 'dxn': dxn * conv, 'dy': dy * conv, 'dz': dz * conv}
    mesh_sizes = {'neuron_mesh_size': nmesh * geometrical_params['rad_axon'],
                  'probe_mesh_size': pmesh * geometrical_params['rad_axon'],
                  'rest_mesh_size': rmesh * geometrical_params['rad_axon']}
    if neurontype == 'sphere':
        neuron = SphereNeuron(geometrical_params)
    elif neurontype == 'mainen':
        neuron = MainenNeuron(geometrical_params)

    probe_x = probetip[0]*conv
    probe_y = probetip[1]*conv
    probe_z = probetip[2]*conv

    print('Probe tip: ', probe_x, probe_y, probe_z)

    if probetype == 'cylinder':
        probe = CylinderProbe({'rad_probe': rad*conv, 'probe_x': probe_x, 'probe_y': probe_y, 'probe_z': probe_z})
    elif probetype == 'box':
        probe = BoxProbe({'probe_dx': 20*conv, 'probe_dy': 20*conv,
                          'probe_x': probe_x, 'probe_y': probe_y, 'probe_z': probe_z})
    elif probetype == 'wedge':
        contact_pts = [(0, h*conv) for h in np.linspace(10, 100, 5)]
        probe = WedgeProbe({'alpha': pi / 4,
                            'probe_x': probe_x, 'probe_y': probe_y, 'probe_z': probe_z,
                            'probe_width': 50*conv, 'probe_thick': 30*conv,
                            'contact_points': contact_pts, 'contact_rad': 5*conv})
    elif probetype == 'fancy':
        # Angle in radians
        probe = FancyProbe({'probe_x': probe_x, 'probe_y': probe_y, 'probe_z': probe_z,
                            'with_contacts': 1, 'rot_angle': np.deg2rad(rot)})
    elif probetype == 'pixel':
        # Angle in radians
        probe = PixelProbe({'probe_x': probe_x, 'probe_y': probe_y, 'probe_z': probe_z,
                            'with_contacts': 1, 'rot_angle': np.deg2rad(rot)})

    if not hide_neuron:
        mesh_name = neurontype + '_' + probetype + '_' + str(probetip[0]) + '_' + str(probetip[1]) + '_' \
                    + str(probetip[2]) + '_coarse_' + str(coarse) + '_box_' + str(box) + '_rot_' + str(rot) + '_rad_' + str(rad)
    else:
        mesh_name = 'noneuron' + '_' + probetype + '_' + str(probetip[0]) + '_' + str(probetip[1]) + '_' \
                    + str(probetip[2]) + '_coarse_' + str(coarse) + '_box_' + str(box) + '_rot_' + str(
            rot) + '_rad_' + str(rad)
    
    if not os.path.isdir(join(root, probetype)):
        os.mkdir(join(root, probetype))
    if not os.path.isdir(join(root, probetype, mesh_name)):
        os.mkdir(join(root, probetype, mesh_name))

    if not hide_neuron:
        fname_wprobe = join(root, probetype, mesh_name, mesh_name + '_wprobe')
        out_wprobe = geofile(neuron, mesh_sizes, probe=probe, file_name=fname_wprobe, hide_neuron=hide_neuron)
        fname_noprobe = join(root, probetype, mesh_name, mesh_name + '_noprobe')
        out_noprobe = geofile(neuron, mesh_sizes, probe=None, file_name=fname_noprobe, hide_neuron=hide_neuron)
    else:
        print('Mesh name', mesh_name)
        if not os.path.isdir(join(root, 'noneuron', probetype, mesh_name)):
            os.makedirs(join(root, 'noneuron', probetype, mesh_name))
            print('Created ', join(root, 'noneuron', probetype, mesh_name))
        fname_wprobe = join(root, 'noneuron', probetype, mesh_name, mesh_name + '_wprobe')
        out_wprobe = geofile(neuron, mesh_sizes, probe=probe, file_name=fname_wprobe, hide_neuron=hide_neuron)


    sys.exit
    import subprocess
    if show:
        subprocess.call(['gmsh %s' % out_wprobe], shell=True)
        if not hide_neuron:
            subprocess.call(['gmsh %s' % out_noprobe], shell=True)
    else:
        subprocess.call(['gmsh -3 %s' % out_wprobe], shell=True)
        if not hide_neuron:
            subprocess.call(['gmsh -3 %s' % out_noprobe], shell=True)

    out_msh_wprobe = out_wprobe[:-3] + 'msh'
    if not hide_neuron:
        out_msh_noprobe = out_noprobe[:-3] + 'msh'

    if return_f:
        sys.exit(out_msh_noprobe + ' ' + out_msh_wprobe)

