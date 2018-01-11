from math import sqrt
import numpy as np


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

    def __str__(self):
        '''String encoding used to loopup correct geo file'''
        raise NotImplementedError

    
class Probe(object):
    '''Base class for gmsh probe'''
    def __init__(self, params):
        '''Check that the construction parameters are sane'''
        assert self.sane_inputs(params)
        self.params = params

    def control_points(self):
        '''Control points of the probe'''
        raise NotImplementedError

    def __str__(self):
        '''String encoding used to loopup correct geo file'''
        raise NotImplementedError

# -------------------------------------------------------------------

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
            return x[0]**2 + x[1]**1 < self.rad_dend**2 + tol
        # Middle sphere
        elif -base_axon - tol < x[2] < base_dend + tol:
            return x[0]**2 + x[1]**1 < self.rad_soma**2 + tol
        # Bottom cylinder
        elif bot_axon - tol < x[2] < -base_axon + tol:
            return x[0]**2 + x[1]**1 < self.rad_axon**2 + tol
        else:
            return False

# --------------------------------------------------------------------
        
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
