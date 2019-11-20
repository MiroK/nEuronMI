from .utils import as_namedtuple, has_positive_values, link_surfaces, entity_dim, second
from .gmsh_primitives import Box
from .baseprobe import Probe
import numpy as np


class Neuropixels24Probe(Probe):
    '''
    Z-axis aligned specified by (x, y, z) of the endpoint, rotation 
    angle around z-axis and length.
    '''
    _defaults = {
        'tip_x': 50,
        'tip_y': 0,
        'tip_z': -200,
        'angle': 0,  # In radians as gmsh want it!
        'length': 1000,
        'width': 20,
        'contact_rad': 12
        }
    
    def __init__(self, params=None):
        Probe.__init__(self, params)

        params = as_namedtuple(self.params_cm)
        
        # Control points outline the shape
        # ----
        # |  |
        # |  |
        #  \/
        tip_x, tip_y, tip_z = params.tip_x, params.tip_y, params.tip_z
        length, width, angle = params.length, params.width, params.angle
        # NOTE: 150, 35, here are unitless. Front plane

        # Rotate around z axis passing through (tip_x, tip_y, tip_z)
        a = np.array([tip_x - 0.5*width, tip_y, tip_z])
        b0 = np.array([tip_x - 0.5*width, tip_y+35, tip_z+150])
        c0 = np.array([tip_x - 0.5*width, tip_y+35, tip_z+length])
        c1 = np.array([tip_x - 0.5*width, tip_y-35, tip_z+length])
        b1 = np.array([tip_x - 0.5*width, tip_y-35, tip_z+150])

        front = [a, b0, c0, c1, b1] 
        # Make rear plane, move away in x
        back = list(map(lambda x: x+width*np.array([1, 0, 0]), front))
        A, B0, C0, C1, B1 = back
        self._control_points = np.row_stack(front + back)
                               
        # Setup bounding box

        ll = np.array([tip_x - 0.5*width, tip_y-35, tip_z])
        ur = np.array([tip_x + 0.5*width, tip_y+35, tip_z+length])
        
        self._bbox = Box(self.rotate(ll), self.rotate(ur) - self.rotate(ll))
        # Keep the unrotated on to do contains
        self._rbbox = Box(ll, ur - ll)

        # Centers of electrodes
        y_shifts = [2+12+6, -1-6, 1+6, -2-12-6]  # Again somethind hardcoded
        z_shifts = [6, 6, 26, 26]
        nrows = 5
        
        contacts = []
        # NOTE: these points are not rotated!
        for y_shift, z_shift in zip(y_shifts, z_shifts):
            for i in range(nrows):
                cx = tip_x - 0.5*width;
                cy = tip_y + y_shift;
                cz = tip_z + 150 + i*40 + z_shift;
                contacts.append([cx, cy, cz])
        self._contacts = np.array(contacts)

        # Not just contacts
        self._surfaces = dict(('contact_%d' % i, c) for i, c in enumerate(self._contacts))
        # By matching points we can also get \/. The remaining surfaces aren't known
        # at this points because the probe will be chopped
        self._surfaces['outline_tip_ymin'] = np.mean(np.row_stack([a, A, b0, B0]), axis=0)
        self._surfaces['outline_tip_ymax'] = np.mean(np.row_stack([a, A, b1, B1]), axis=0)
        self._surfaces['outline_front'] = a
        self._surfaces['outline_back'] = A
        self._surfaces['outline_ymin'] = 0.5*(b0 + B0)
        self._surfaces['outline_ymax'] = 0.5*(b1 + B1)

    def check_geometry_parameters(self, params):
        assert set(params.keys()) == set(Neuropixels24Probe._defaults.keys()), (set(params.keys()), set(Neuropixels24Probe._defaults.keys()))
        # Ignore center
        assert all(params[k] > 0 for k in ('width', 'length', 'contact_rad'))

    def rotate(self, x, a=None):
        '''Transformation that rotates the probe'''
        if a is None:
            a = self.params_cm['angle']
            
        R = np.array([[np.cos(a), -np.sin(a), 0],
                      [np.sin(a), np.cos(a), 0],
                      [0, 0, 1.]])

        x0 = np.array([self.params_cm[k] for k in ('tip_x', 'tip_y', 'tip_z')])

        return x0 + R.dot(x-x0)
        
    def contains(self, point, tol):
        '''Is point inside shape?'''
        return self._rbbox.contains(self.rotate(point, -self.params_cm['angle']), tol=tol)
    
    def as_gmsh(self, model, tag=-1):
        '''Add shape to model in terms of factory(gmsh) primitives'''
        factory = model.occ

        contact_rad, width = self.params_cm['contact_rad'], self.params_cm['width']

        volumes = []
        # Addd a bit shorter boxes will given electrode surfaces
        for (cx, cy, cz) in self._contacts:
            # Unrotated
            vid = factory.addBox(cx, cy-contact_rad/2, cz-contact_rad/2, 0.8*width, contact_rad, contact_rad)
            volumes.append((3, vid))
        factory.synchronize()

        points = self._control_points[:len(self._control_points)//2]
        # Outline of the probe
        p_idx = [factory.addPoint(*p) for p in points]
        # Trace out
        l_idx = [factory.addLine(*l) for l in zip(p_idx, p_idx[1:] + [p_idx[0]])]
        # Front face
        loop = factory.addCurveLoop(l_idx) 
        surf = factory.addPlaneSurface([loop])
        # Extrude
        entities = factory.extrude([(2, surf)], dx=width, dy=0, dz=0)
        bounding_volume = entity_dim(entities, dim=3)

        factory.synchronize()
        # Combine
        entities = factory.fragment(bounding_volume, volumes)
        probe_volumes = entity_dim(entities, 3)  # [(3, id), ---]
        
        # To rotate the entire probe
        angle = self.params_cm['angle']  
        tip_x, tip_y, tip_z = (self.params_cm[k] for k in ('tip_x', 'tip_y', 'tip_z'))
        factory.rotate(probe_volumes, tip_x, tip_y, tip_z, 0, 0, 1, angle)
        
        probe_tags = list(map(second, probe_volumes))
        
        return probe_tags

    def link_surfaces(self, model, tags, links, box, tol=1E-10):
        '''Account for possible cut and shift of center of mass of face'''
        # Account for probe rotation in tagging
        for surf, point in self._surfaces.items():
            self._surfaces[surf] = self.rotate(point)

        links = link_surfaces(model, tags, self, links=links, tol=tol)
        # NOTE: as we chop the by box, the wall won't be found with the above metric
        # So we don't match on Z
        metric = lambda x, y: np.sqrt(((y - x)[:, 0])**2 + ((y - x)[:, 1])**2)
        links = link_surfaces(model, tags, self, links=links, metric=metric, tol=tol)

        return links

# --------------------------------------------------------------------
