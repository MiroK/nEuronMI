from math import sqrt, tan, atan
import numpy as np

# TODO  make more modular: create extra folders - neurons/ probes/ and have one file per probe. Easier to extend
#  Then import from neurons import *, from probes import *
#  Same for geo_codes. We should be able to modularly construct a mesh


class BBox(object):
    '''Box specified by lower left coord and extents'''
    def __init__(self, x, dx):
        self.x = np.asarray(x)
        self.y = self.x + np.asarray(dx)

    def is_inside(self, p, tol=1E-13):
        '''Point in the box'''
        x, y = self.x, self.y

        return all(xi - tol < pi < yi + tol for xi, pi, yi in zip(x, p, y))


class Neuron(object):
    '''Base class for gmsh neuron'''
    def __init__(self, params):
        '''Check that the construction parameters are sane'''
        assert self.sane_inputs(params)

        self.geom_bbox = self.compute_bbox()
        self.params = params
        
    def compute_bbox(self, params):
        '''Bounding box of the geomery'''
        raise NotImplementedError
        
    def is_inside(self, x, tol=1E-13):
        '''Is x conained in the neuron'''
        raise NotImplementedError

    def is_inside_bbox(self, x, tol=1E-13):
        '''Is x inside the bounding box of the geometry'''
        return self.geom_bbox.is_inside(x, tol)

    def definitions(self, is_masked=False):
        '''Addtional code for geo file'''
        return 'is_masked=%g;' % int(is_masked)

    def __str__(self):
        '''String encoding used to loopup correct geo file'''
        raise NotImplementedError

    def plot(self, ax, npoints=50000):
        '''Plot the neuron in x[i]=c by bombarding with npoints'''
        #
        #from mpl_toolkits.mplot3d import Axes3D
        #import matplotlib.pyplot as plt
        #
        x, y = self.geom_bbox.x, self.geom_bbox.y
        dx = y - x
        pts = np.random.rand(npoints, 3)

        pts = x + pts*dx
        pts = pts[map(self.is_inside, pts)].T

        ax.scatter(pts[0], pts[1], pts[2])

        return ax

    
class Probe(object):
    '''Base class for gmsh probe'''
    def __init__(self, params):
        '''Check that the construction parameters are sane'''
        assert self.sane_inputs(params)
        self.params = params

    def control_points(self):
        '''Control points of the probe'''
        raise NotImplementedError

    def definitions(self, neuron):
        '''Addtional code for geo file'''
        return ''

    def __str__(self):
        '''String encoding used to loopup correct geo file'''
        raise NotImplementedError

    
##########
# NEURONS
##########

#TODO rename BASNeuron
class SphereNeuron(Neuron):
    '''
    Neuron made of a spherical soma at 0, 0, 0 with 2 cylinders in z
    direction which represent dendrite/axon
    '''
    
    def sane_inputs(self, params):
        all_params = ('rad_soma',
                      'rad_dend', 'length_dend',
                      'rad_axon', 'length_axon',
                      'dxp', 'dxn', 'dy', 'dz')

        assert all(key in params for key in all_params)

        assert all(params[key] > 0 for key in all_params)

        assert all((params['rad_soma'] > params['rad_dend'],
                    params['rad_soma'] > params['rad_axon']))
        for key in all_params: setattr(self, key, params[key])

        return True

    def compute_bbox(self):
        '''Bounding box of the geomery'''
        x = -self.rad_soma - self.dxn;
        dx = self.rad_soma + self.dxp - x;

        y = -self.rad_soma - self.dy;
        dy = 2*abs(y);

        base_dend = sqrt(self.rad_soma**2 - self.rad_dend**2);
        base_axon = sqrt(self.rad_soma**2 - self.rad_axon**2);

        z0 = -base_axon - self.length_axon - self.dz;
        z1 = base_dend + self.length_dend + self.dz;
        dz = z1 - z0;
        
        return BBox([x, y, z0], [dx, dy, dz])

    def __str__(self): return 'sphere_neuron'

    def is_inside(self, x, tol=1E-13):
        '''Is x conained in the neuron'''
        base_dend = sqrt(self.rad_soma**2 - self.rad_dend**2);
        base_axon = sqrt(self.rad_soma**2 - self.rad_axon**2);

        bot_axon = -base_axon - self.length_axon;
        top_dend = base_dend + self.length_dend

        # Top cylinder
        if base_dend - tol < x[2] < top_dend + tol:
            return x[0]**2 + x[1]**2 < self.rad_dend**2 + tol
        # Middle sphere
        elif -base_axon - tol < x[2] < base_dend + tol:
            return x[0]**2 + x[1]**2 < self.rad_soma**2 + tol
        # Bottom cylinder
        elif bot_axon - tol < x[2] < -base_axon + tol:
            return x[0]**2 + x[1]**2 < self.rad_axon**2 + tol
        else:
            return False

#TODO add mechanism to use many neurons ad different positions
class SphereNeurons2(SphereNeuron):
    '''Two sphere neurons which are dist in x direction apart'''    
    def sane_inputs(self, params):
        all_params = ('rad_soma',
                      'rad_dend', 'length_dend',
                      'rad_axon', 'length_axon',
                      'dxp', 'dxn', 'dy', 'dz',
                      'dist')

        assert all(key in params for key in all_params)

        assert all(params[key] > 0 for key in all_params)

        assert all((params['rad_soma'] > params['rad_dend'],
                    params['rad_soma'] > params['rad_axon']))
        for key in all_params: setattr(self, key, params[key])

        return True

    def compute_bbox(self):
        '''Bounding box of the geomery'''
        x = -self.rad_soma - self.dxn;
        dx = self.rad_soma + self.dxp + self.dist - x;

        y = -self.rad_soma - self.dy;
        dy = 2*abs(y);

        base_dend = sqrt(self.rad_soma**2 - self.rad_dend**2);
        base_axon = sqrt(self.rad_soma**2 - self.rad_axon**2);

        z0 = -base_axon - self.length_axon - self.dz;
        z1 = base_dend + self.length_dend + self.dz;
        dz = z1 - z0;
        
        return BBox([x, y, z0], [dx, dy, dz])

    def __str__(self): return 'sphere_neurons2'

    def is_inside(self, x, tol=1E-13):
        '''Is x conained in the neuron'''
        return SphereNeuron.is_inside(self, x, tol) or SphereNeuron.is_inside(self, x-np.array([self.dist, 0, 0]), tol)

#TODO rename TaperedNeuron
class MainenNeuron(Neuron):
    '''
    Neuron made of a spherical soma at 0, 0, 0 with 2 cylinders in z
    direction which represent dendrite/axon. Between soma-axon, soma-dend
    there are hilux segments which are modeled as cones/cylinders
    '''
    
    def sane_inputs(self, params):
        all_params = ('rad_soma',
                      'rad_dend', 'length_dend',
                      'rad_axon', 'length_axon',
                      'rad_hilox_d', 'length_hilox_d',
                      'rad_hilox_a', 'length_hilox_a',
                      'dxp', 'dxn', 'dy', 'dz')

        assert all(key in params for key in all_params)

        assert all(params[key] > 0 for key in all_params)

        assert all((params['rad_soma'] > params['rad_hilox_d'] >= params['rad_dend'],
                    params['rad_soma'] > params['rad_hilox_a'] >= params['rad_axon']))
        for key in all_params: setattr(self, key, params[key])

        return True

    def compute_bbox(self):
        '''Bounding box of the geomery'''
        x = -self.rad_soma - self.dxn;
        dx = self.rad_soma + self.dxp - x;

        y = -self.rad_soma - self.dy;
        dy = 2*abs(y);

        base_d = sqrt(self.rad_soma**2 - self.rad_hilox_d**2);
        base_a = sqrt(self.rad_soma**2 - self.rad_hilox_a**2);

        z0 = -base_a - self.length_hilox_a - self.length_axon - self.dz;
        z1 = base_d + self.length_hilox_d + self.length_dend + self.dz;
        dz = z1 - z0;
        
        return BBox([x, y, z0], [dx, dy, dz])

    def __str__(self): return 'mainen_neuron'

    def is_inside(self, x, tol=1E-13):
        '''Is x conained in the neuron'''
        base_hilox_d = sqrt(self.rad_soma**2 - self.rad_hilox_d**2);
        base_hilox_a = sqrt(self.rad_soma**2 - self.rad_hilox_a**2);
                
        base_dend = base_hilox_d + self.length_hilox_d
        base_axon = base_hilox_a + self.length_hilox_a

        top_dend = base_dend + self.length_dend
        bot_axon = -base_axon - self.length_axon

        dr_d = self.rad_hilox_d - self.rad_dend
        dr_a = self.rad_hilox_a - self.rad_axon
        
        # Top cylinder
        if base_dend - tol < x[2] < top_dend + tol:
            return x[0]**2 + x[1]**2 < self.rad_dend**2 + tol
        # Then cone
        elif base_hilox_d - tol < x[2] < base_dend + tol:
            rad = self.rad_dend + dr_d*abs(base_dend-x[2])/self.length_hilox_d
            return x[0]**2 + x[1]**2 < rad**2 + tol
        # Middle sphere
        elif -base_hilox_a - tol < x[2] < base_hilox_d + tol:
            return x[0]**2 + x[1]**2 < self.rad_soma**2 + tol
        # Then cone
        elif -base_axon - tol < x[2] < -base_hilox_a + tol:
            rad = self.rad_axon + dr_a*abs(-base_axon-x[2])/self.length_hilox_a
            return x[0]**2 + x[1]**2 < rad**2 + tol
        # Bottom cylinder
        elif bot_axon - tol < x[2] < -base_axon + tol:
            return x[0]**2 + x[1]**2 < self.rad_axon**2 + tol
        else:
            return False

        
class MainenNeurons2(MainenNeuron):
    '''2 Mainen neuons which are dist apart in x direction'''

    def sane_inputs(self, params):
        all_params = ('rad_soma',
                      'rad_dend', 'length_dend',
                      'rad_axon', 'length_axon',
                      'rad_hilox_d', 'length_hilox_d',
                      'rad_hilox_a', 'length_hilox_a',
                      'dxp', 'dxn', 'dy', 'dz',
                      'dist')

        assert all(key in params for key in all_params)

        assert all(params[key] > 0 for key in all_params)

        assert all((params['rad_soma'] > params['rad_hilox_d'] >= params['rad_dend'],
                    params['rad_soma'] > params['rad_hilox_a'] >= params['rad_axon']))
        for key in all_params: setattr(self, key, params[key])

        return True

    def compute_bbox(self):
        '''Bounding box of the geomery'''
        x = -self.rad_soma - self.dxn;
        dx = self.rad_soma + self.dxp + self.dist - x;

        y = -self.rad_soma - self.dy;
        dy = 2*abs(y);

        base_d = sqrt(self.rad_soma**2 - self.rad_hilox_d**2);
        base_a = sqrt(self.rad_soma**2 - self.rad_hilox_a**2);

        z0 = -base_a - self.length_hilox_a - self.length_axon - self.dz;
        z1 = base_d + self.length_hilox_d + self.length_dend + self.dz;
        dz = z1 - z0;
        
        return BBox([x, y, z0], [dx, dy, dz])

    def __str__(self): return 'mainen_neurons2'

    def is_inside(self, x, tol=1E-13):
        '''Is x conained in the neuron'''
        return MainenNeuron.is_inside(self, x, tol) or MainenNeuron.is_inside(self, x-np.array([self.dist, 0, 0]), tol)

        
#########
# PROBES
######### 

#TODO rename MicrowireProbe
class CylinderProbe(Probe):
    '''Probe shaped like a cylinder.'''
    
    def sane_inputs(self, params):
        all_params = ('rad_probe', 'probe_x', 'probe_y', 'probe_z')
        
        assert all(key in params for key in all_params)
        assert params['rad_probe'] > 0
        
        for key in all_params: setattr(self, key, params[key])

        return True

    def control_points(self):
        '''Control points of the probe'''
        x0 = np.array([self.probe_x, self.probe_y, self.probe_z])
        return (x0 + np.array([self.rad_probe, 0, 0]),
                x0 + np.array([-self.rad_probe, 0, 0]),
                x0 + np.array([0, self.rad_probe, 0]),
                x0 + np.array([0, -self.rad_probe, 0])) 

    def __str__(self): return 'cylinder_probe'


class BoxProbe(Probe):
    '''A box shape with z plane crossection as rectengle.'''
    
    def sane_inputs(self, params):
        all_params = ('probe_x', 'probe_y', 'probe_z', 'probe_dx', 'probe_dy')
        
        assert all(key in params for key in all_params)
        assert params['probe_dx'] > 0 and params['probe_dy'] > 0

        for key in all_params: setattr(self, key, params[key])

        return True

    def control_points(self):
        '''Control points of the probe'''
        x0 = np.array([self.probe_x, self.probe_y, self.probe_z])
        return (x0 + 0.5*np.array([self.probe_dx, self.probe_dy, 0]),
                x0 + 0.5*np.array([-self.probe_dx, self.probe_dy, 0]),
                x0 + 0.5*np.array([-self.probe_dx, -self.probe_dy, 0]),
                x0 + 0.5*np.array([self.probe_dx, -self.probe_dy, 0]))

    def __str__(self): return 'box_probe'

#TODO remove
class WedgeProbe(Probe):
    '''
    In the plane normal to x-axis the probe's crossection is 
    <--> w    

    |  | 
    |  |
     \/
   
    ^z 
    |
    -->y
    with the tip at probe_x, probe_y, probe_z. Other parameters of the
    probe are probe_width and probe_thick (ness) and the tip angle 
    alpha. An optinal list of y,z coordinates  specigies positions of 
    the conducting surfaces which are circle of radii contact_rad.
    '''
    def sane_inputs(self, params):

        all_params = ('probe_x', 'probe_y', 'probe_z', 'alpha', 'probe_width', 'probe_thick')
        
        assert all(key in params for key in all_params)
        assert params['probe_width'] > 0 and params['probe_thick'] > 0
        assert 0 < params['alpha'] < np.pi
        
        for key in all_params: setattr(self, key, params[key])

        self.probe_dz = abs(self.probe_width/2/tan(self.alpha/2))
        
        # Contacts are optional
        contacts = params.get('contact_points', [])
        # With contacts there must be a radius and poins must be in the probe. 
        if contacts:
            assert params['contact_rad'] > 0
            
            rad = params['contact_rad']
            # The probes don't overlap
            for i, c0 in enumerate(contacts):
                for c1 in contacts[i+1:]:
                    # At least two radii away
                    assert (c0[0]-c1[0])**2 + (c0[1]-c1[1])**2 > (2*rad)**2

            # The contacts are well within the probe. Note that this does
            # not mean that all the contacts will be in the final domain
            # as the probe length is determined by the bbox which is not
            # known a this point.
            # _|_
            #  |
            contact_control_pts = lambda (y, z): [(y+rad, z+rad),
                                                  (y+rad, z-rad),
                                                  (y-rad, z+rad),
                                                  (y-rad, z-rad)]

            for y, z in sum(map(contact_control_pts, contacts), []):
                assert z > self.probe_z, (z, self.probe_z)
                # Check the tip, wedge
                if z < self.probe_z + self.probe_dz:
                    assert atan(abs(y)/(z - self.probe_z)) < self.alpha/2,\
                        (atan(abs(y)/(z - self.probe_z)), self.alpha/2)
                # The rectangular probe
                else:
                    assert abs(y - self.probe_y) < self.probe_width/2

            # Points are only needed for the code. No need to keep around
            # for loenger
            y_pts = map(lambda p: p[0], contacts)
            code_y = 'contact_loc_y[] = {%s};' % (', '.join(map(str, y_pts)))

            z_pts = map(lambda p: p[1], contacts)
            code_z = 'contact_loc_z[] = {%s};' % (', '.join(map(str, z_pts)))

            self.contact_points_code = '\n'.join([code_y, code_z])

            # Remove conatct_points from params so that they are not
            # added to header defs in code gen
            del params['contact_points']
        else:
            if 'contact_points' in params: del params['contact_points']
            self.contact_points_code = 'contact_loc_z[] = {};'


        return True

    def control_points(self):
        '''Control points of the probe'''
        #  __
        #  \/
        points = np.array([[self.probe_x-self.
                            probe_thick/2,
                            self.probe_y-self.probe_width/2,
                            self.probe_z + self.probe_dz],
                           [self.probe_x-self.probe_thick/2,
                            self.probe_y,
                            self.probe_z],
                           [self.probe_x-self.probe_thick/2,
                            self.probe_y+self.probe_width/2,
                            self.probe_z + self.probe_dz]])
        # Add the extruded points
        points = np.r_[points, points + np.array([self.probe_thick, 0, 0])]

        return points

    def definitions(self, neuron):
        '''Code for contact points defition and probe top from bbox'''
        return '\n'.join([self.contact_points_code,
                          'probe_top = %g + 2;' % neuron.geom_bbox.y[2]])
    
    def __str__(self): return 'wedge_probe'

#TODO rename Neuronexus32Probe
class FancyProbe(Probe):
    '''The realistic probe. Everything is hardcoded...'''
    def sane_inputs(self, params):

        all_params = ('probe_x', 'probe_y', 'probe_z')
        
        assert all(key in params for key in all_params)
        
        for key in all_params: setattr(self, key, params[key])
        # Additional
        self.conv = 1E-4
        self.probe_thick = 15*self.conv
        self.contact_rad = 7.5*self.conv
        self.with_contacts = int(params.get('with_contacts', 0))

        self.rot_angle = float(params.get('rot_angle', 0.))
        self.cz = 62*self.conv + sqrt(22*22-18*18)*self.conv + 9*25*self.conv

        return True
        
    def control_points(self):
        '''Control points of the probe'''
        # These define the tip
        #         |     cz
        #       _/      62
        #      /
        #    /
        #    0 31 57
        points = np.array([[self.probe_x - self.probe_thick/2., self.probe_y, self.probe_z],
                           [self.probe_x - self.probe_thick/2., self.probe_y+31*self.conv, self.probe_z+62*self.conv],
                           [self.probe_x - self.probe_thick/2., self.probe_y+57*self.conv, self.probe_z+self.cz],
                           [self.probe_x - self.probe_thick/2., self.probe_y-31*self.conv, self.probe_z+62*self.conv],
                           [self.probe_x - self.probe_thick/2., self.probe_y-57*self.conv, self.probe_z+self.cz]])
        
        points = np.r_[points, points + np.array([self.probe_thick, 0, 0])]

        return points

    def definitions(self, neuron):
        '''Code for contact points defition and probe top from bbox'''

        return '\n'.join(['conv = %g;' % self.conv, 
                          'probe_thick = %g;' % self.probe_thick,
                          'contact_rad = %g;' % self.contact_rad,
                          'with_contacts = %d;' % self.with_contacts,
                          'rot_angle = %g;' % self.rot_angle,
                          'probe_top = %g + %g;' % (neuron.geom_bbox.y[2], 200*self.conv)])
    
    def __str__(self): return 'fancy_probe'

#TODO rename Neuropixels24Probe
class PixelProbe(Probe):
    '''The realistic probe. Everything is hardcoded...'''

    def sane_inputs(self, params):
        all_params = ('probe_x', 'probe_y', 'probe_z')

        assert all(key in params for key in all_params)

        for key in all_params: setattr(self, key, params[key])
        # Additional
        self.conv = 1E-4
        self.probe_thick = 20 * self.conv
        self.contact_rad = 12 * self.conv
        self.with_contacts = int(params.get('with_contacts', 0))

        self.rot_angle = float(params.get('rot_angle', 0.))

        return True

    def control_points(self):
        '''Control points of the probe'''
        # These define the tip
        #         |
        #       _/      150
        #      /
        #    /
        #    0 35
        points = np.array([[self.probe_x - self.probe_thick / 2., self.probe_y, self.probe_z],
                           [self.probe_x - self.probe_thick / 2., self.probe_y + 35 * self.conv,
                            self.probe_z + 150 * self.conv],
                           [self.probe_x - self.probe_thick / 2., self.probe_y - 35 * self.conv,
                            self.probe_z + 150 * self.conv]])

        points = np.r_[points, points + np.array([self.probe_thick, 0, 0])]

        return points

    def definitions(self, neuron):
        '''Code for contact points defition and probe top from bbox'''

        return '\n'.join(['conv = %g;' % self.conv,
                          'probe_thick = %g;' % self.probe_thick,
                          'contact_rad = %g;' % self.contact_rad,
                          'with_contacts = %d;' % self.with_contacts,
                          'rot_angle = %g;' % self.rot_angle,
                          'probe_top = %g + %g;' % (neuron.geom_bbox.y[2], 500 * self.conv)])

    def __str__(self): return 'pixel_probe'