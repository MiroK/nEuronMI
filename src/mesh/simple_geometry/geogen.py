import hashlib, time, os
from os.path import join


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

def geofile(neuron, sizes, file_name='', probe=None):
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
    neuron_defs = neuron.definitions()
    
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

    print 'Generated geo file', geo_file
    print 'Inspect with Gmsh or run e.g. `gmsh -3 $GEOFILE` to generate mesh'

    with open(geo_file, 'w') as f: f.write(code)

    return geo_file

# -------------------------------------------------------------------

if __name__ == '__main__':
    from shapes import SphereNeuron, MainenNeuron
    from shapes import CylinderProbe, BoxProbe, WedgeProbe, FancyProbe
    from math import pi
    import sys

    if '-f' in sys.argv:
        pos = sys.argv.index('-f')
        fname = sys.argv[pos + 1]
    else:
        fname=''
    if '-simple' in sys.argv:
        simple=True
    else:
        simple=False
    if '-probetype' in sys.argv:
        pos = sys.argv.index('-probetype')
        probetype = sys.argv[pos + 1]
    else:
        probetype='cylinder'
    if '-neurontype' in sys.argv:
        pos = sys.argv.index('-neurontype')
        neurontype = sys.argv[pos + 1]
    else:
        neurontype='sphere'
    if '-probetip' in sys.argv:
        pos = sys.argv.index('-probetip')
        probetip = list(sys.argv[pos + 1])
    else:
        probetip=[50, 0, 0]

    if len(sys.argv) == 1:
        print 'Generate GEO and msh files with and without probe. \n   -f : filename'\
              '\n   -simple : simpler mesh (coarser cells - larger neuron)' \
              '\n   -probetype : cylinder (default) - box - wedge - fancy\n   -neurontype : sphere (default) - mainen' \
              '\n   -probetip : [x y z] of probe tip (in um)'
        raise Exception('Indicate mesh argumets')
    
    conv=1E-4

    ########################
    # SIMPLE
    ########################
    if simple:
        geometrical_params = {'rad_soma': 30 * conv, 'rad_dend': 10 * conv, 'rad_axon': 6 * conv,
                              'length_dend': 400 * conv, 'length_axon': 200 * conv, 'rad_hilox_d': 16 * conv,
                              'length_hilox_d': 20 * conv, 'rad_hilox_a': 10 * conv, 'length_hilox_a': 10 * conv,
                              'dxp': 80 * conv, 'dxn': 80 * conv, 'dy': 50 * conv, 'dz': 40 * conv}
        mesh_sizes = {'neuron_mesh_size': 2 * geometrical_params['rad_axon'],
                      'probe_mesh_size': 2 * geometrical_params['rad_axon'],
                      'rest_mesh_size': 4 * geometrical_params['rad_axon']}
    #####################################
    # DETAILED
    #####################################
    else:
        geometrical_params = {'rad_soma': 15 * conv, 'rad_dend': 4 * conv, 'rad_axon': 1 * conv,
                              'length_dend': 400 * conv,
                              'length_axon': 200 * conv, 'rad_hilox_d': 8 * conv, 'length_hilox_d': 20 * conv,
                              'rad_hilox_a': 4 * conv,
                              'length_hilox_a': 10 * conv, 'dxp': 100 * conv, 'dxn': 80 * conv, 'dy': 80 * conv,
                              'dz': 40 * conv}
        mesh_sizes = {'neuron_mesh_size': 2 * geometrical_params['rad_axon'],
                      'probe_mesh_size': 4 * geometrical_params['rad_axon'],
                      'rest_mesh_size': 4 * geometrical_params['rad_axon']}

    if neurontype == 'sphere':
        neuron = SphereNeuron(geometrical_params)
    elif neuron == 'mainen':
        neurontype = MainenNeuron(geometrical_params)

    probe_x = probetip[0]*conv
    probe_y = probetip[1]*conv
    probe_z = probetip[2]*conv

    print probe_x, probe_y, probe_z

    if probetype == 'cylinder':
        probe = CylinderProbe({'rad_probe': 15*conv, 'probe_x': probe_x, 'probe_y': probe_y, 'probe_z': probe_z})
    elif probetype == 'box':
        probe = BoxProbe({'probe_dx': 20*conv, 'probe_dy': 20*conv,
                          'probe_x': probe_x, 'probe_y': probe_y, 'probe_z': probe_z})
    elif probetype == 'wedge':
        contact_pts = [(0, 0.7), (0, 1.0), (0, 1.3), (0, 0.3)]
        probe = WedgeProbe({'alpha': pi / 4,
                            'probe_x': probe_x, 'probe_y': probe_y, 'probe_z': probe_z,
                            'probe_width': 50*conv, 'probe_thick': 30*conv,
                            'contact_points': contact_pts, 'contact_rad': 5*conv})
    elif probetype == 'fancy':
        probe = FancyProbe({'probe_x': probe_x, 'probe_y': probe_y, 'probe_z': probe_z, 'with_contacts': 1})

    if not os.path.isdir(probetype):
        os.mkdir(probetype)

    fname_wprobe = join(probetype, neurontype + '_' + probetype + '_' + str(probetip[0]) + '_' + str(probetip[1]) + '_'
                        + str(probetip[2]) + '_wprobe')
    out_wprobe = geofile(neuron, mesh_sizes, probe=probe, file_name=fname_wprobe)
    fname_noprobe = join(probetype, neurontype + '_' + probetype + '_' + str(probetip[0]) + '_' + str(probetip[1]) + '_'
                         + str(probetip[2]) + '_noprobe')
    out_noprobe = geofile(neuron, mesh_sizes, probe=None, file_name=fname_noprobe)

    import subprocess
    subprocess.call(['gmsh %s' % out_wprobe], shell=True)
    subprocess.call(['gmsh %s' % out_noprobe], shell=True)

