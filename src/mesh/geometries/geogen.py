import hashlib, time, os


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


def read_code(dispatch, directory='geo_codes'):
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

def geofile(neuron, sizes, probe=None):
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
    geo_file = hashlib.sha1()
    geo_file.update(str(time.time()))
    geo_file = '.'.join([geo_file.hexdigest(), 'GEO'])

    print 'Generated geo file', geo_file
    print 'Inspect with Gmsh or run e.g. `gmsh -3 $GEOFILE` to generate mesh'

    with open(geo_file, 'w') as f: f.write(code)

    return geo_file

# -------------------------------------------------------------------

from shapes import SphereNeuron, MainenNeuron
from shapes import CylinderProbe, BoxProbe, WedgeProbe

neuron = SphereNeuron({'rad_soma': 0.5,
                       'rad_dend': 0.3, 'length_dend': 1,
                       'rad_axon': 0.2, 'length_axon': 1,
                       'dxp': 1.5, 'dxn': 1.25, 'dy': 1.0, 'dz': 0.2})

# neuron = MainenNeuron({'rad_soma': 1,
#                        'rad_hilox_d': 0.4, 'length_hilox_d': 0.3,
#                        'rad_dend': 0.3, 'length_dend': 2,
#                        'rad_hilox_a': 0.3, 'length_hilox_a': 0.4,
#                        'rad_axon': 0.2, 'length_axon': 4,
#                        'dxp': 2.5, 'dxn': 0.5, 'dy': 0.2, 'dz': 0.2})

#probe = CylinderProbe({'rad_probe': 0.2, 'probe_x': 1.5, 'probe_y': 0, 'probe_z': 0})

#probe = BoxProbe({'probe_dx': 0.2, 'probe_dy': 0.2,
#                  'probe_x': 1.5, 'probe_y': 0, 'probe_z': 0})

from math import pi

contact_pts = [(0, 0.7), (0, 1.0), (0, 1.3), (0, 0.3)]
probe = WedgeProbe({'alpha': pi/4,
                    'probe_z': 0, 'probe_x': 1.5, 'probe_y': 0,
                    'probe_width': 0.5, 'probe_thick': 0.3,
                    'contact_points': contact_pts, 'contact_rad': 0.05})
 
sizes = {'neuron_mesh_size': 0.2, 'probe_mesh_size': 0.2, 'rest_mesh_size': 0.4}

out = geofile(neuron, sizes, probe=probe)

import subprocess
subprocess.call(['gmsh %s' % out], shell=True)

# FIXME: the fancy probe
#
#        ellipse soma
#        soma with surface of revolution
#
#        finish the pipeline
