from utils import circle_points, unit_vector
from baseshape import BaseShape
from math import sqrt
import numpy as np
import utils


class Box(BaseShape):
    '''Box is specified by min X, Y, Z coordinates and extents'''
    def __init__(self, p, dx):
        assert len(p) == 3 and len(dx) == 3
        assert np.all(dx > 0)

        self.min_ = p
        self.max_ = p + dx
        self._control_points = [self.min_, self.max_]

        self._bbox = self

        #
        #  D-----C
        #  |\A   |B
        #  d-|---c|
        #   \|    \
        #    a----b
        a = p
        b = p + np.r_[dx[0], 0, 0]
        c = p + np.r_[dx[0], dx[1], 0]
        d = p + np.r_[0, dx[1], 0]
        A, B, C, D = (x + np.array([0, 0, dx[2]]) for x in (a, b, c, d))

        self._center_of_mass = c
        # Centers of 6 faces
        self._surfaces = {'min_x': np.mean((a, A, d, D), axis=0),
                          'max_x': np.mean((c, C, b, B), axis=0),
                          'min_y': np.mean((a, A, b, B), axis=0),
                          'max_y': np.mean((d, D, c, C), axis=0),
                          'min_z': np.mean((a, b, c, d), axis=0),
                          'max_z': np.mean((A, B, C, D), axis=0)}


    def contains(self, point, tol):
        return np.all(np.logical_and(self.min_ - tol < point,
                                     point < self.max_ + tol))

    def as_gmsh(self, model, tag=-1):
        args = np.r_[self.min_, self.max_ - self.min_]
        return model.occ.addBox(*args, tag=tag)

    def link_surfaces(self, model, tags, links, box=None, tol=1E-10):
        '''Account for possible cut and shift of center of mass of face'''
        # This pass is blind
        links = utils.link_surfaces(model, tags, self, links, tol=tol)
        # NOTE: everything is assumed to be z aligned. We might have
        # a cut z plane
        metric = lambda x, y: np.abs((y - x)[:,2])  # But z coord should match
        
        return utils.link_surfaces(model, tags, self, tol=tol, links=links, metric=metric)

    
class Sphere(BaseShape):
    '''Defined by Center + radius'''
    def __init__(self, c, r):
        assert r > 0
        assert len(c) == 3

        self._control_points = np.vstack([circle_points(c, r),  # in x-y plane
                                          np.array([0, 0, -r]),
                                          np.array([0, 0, r])])

        min_ = np.min(self._control_points, axis=0)
        dx = np.max(self._control_points, axis=0) - min_
        self._bbox = Box(min_, dx)

        self.c = c
        self.r = r

        self._center_of_mass = c
        self._surfaces = {'all': c}

    def contains(self, point, tol):
        if not self.bbox_contains(point, tol):
            return False
        return np.linalg.norm(point - self.c) < self.r + tol
               
    def as_gmsh(self, model, tag=-1):
        '''Return volume tag under which shape has been added'''
        args = np.r_[self.c, self.r]
        return model.occ.addSphere(*args, tag=tag)


class Cylinder(BaseShape):
    '''Two endpoints on the centerline and radius'''
    def __init__(self, A, B, r):
        assert r > 0
        assert len(A) == len(B) == 3

        axis = unit_vector(B - A)
        
        self._control_points = np.vstack([circle_points(A, r, axis),  # in x-y plane
                                          circle_points(B, r, axis)])

        min_ = np.min(self._control_points, axis=0)
        dx = np.max(self._control_points, axis=0) - min_
        self._bbox = Box(min_, dx)

        self.A, self.B = A, B
        self.r = r

        self._center_of_mass = A + 0.5*(B-A)
        self._surfaces = {'baseA': A,
                          'baseB': B, 
                          'wall': self._center_of_mass}

    def contains(self, point, tol):
        if not self.bbox_contains(point, tol):
            return False

        A, B, r = self.A, self.B, self.r

        axis = unit_vector(B - A)
        project = np.dot(point-A, axis)

        if abs(project) > np.linalg.norm(B- A):
            return False

        dist = sqrt(abs(np.linalg.norm(point-A)**2 - project**2))

        return dist < r + tol
    
    def as_gmsh(self, model, tag=-1):
        '''Return volume tag under which shape has been added'''
        args = np.r_[self.A, self.B-self.A, self.r]
        return model.occ.addCylinder(*args, tag=tag)


class Cone(BaseShape):
    '''Two endpoints on the centerline and their radii'''
    def __init__(self, A, B, rA, rB):
        assert rA > 0 and rB > 0
        assert len(A) == len(B) == 3

        axis = unit_vector(B - A)
        
        self._control_points = np.vstack([circle_points(A, rA, axis),  # in x-y plane
                                          circle_points(B, rB, axis)])

        min_ = np.min(self._control_points, axis=0)
        dx = np.max(self._control_points, axis=0) - min_
        self._bbox = Box(min_, dx)

        self.A, self.B = A, B
        self.rA, self.rB = rA, rB

        dr = rB - rA
        s = 0.5*(rA**2 + 4*rA*dr/3. + dr**2/2.)/(rA**2 + rA*dr + dr**2/3.)
        assert 0 < s < 1

        self._center_of_mass = A + s*(B-A)
        self._surfaces = {'baseA': A,
                          'baseB': B, 
                          'wall': self._center_of_mass}

        # A useful property of wall is its center of mass
        s = (rA/2. + dr/3.)/(rA + dr/2.)
        self.wall_center_of_mass = A + s*(B-A)

    def contains(self, point, tol):
        if not self.bbox_contains(point, tol):
            return False

        A, B, rA, rB = self.A, self.B, self.rA, self.rB

        axis = unit_vector(B - A)
        project = np.dot(point-A, axis)
        length = np.linalg.norm(B- A)
        if abs(project) > length:
            return False

        dist = sqrt(abs(np.linalg.norm(A-point)**2 - project**2))

        r = (project/length)*(rB-rA) + rA
        return dist < r + tol
    
    def as_gmsh(self, model, tag=-1):
        '''Return volume tag under which shape has been added'''
        args = np.r_[self.A, self.B-self.A, self.rA, self.rB]
        return model.occ.addCone(*args, tag=tag)

# --------------------------------------------------------------------

if __name__ == '__main__':

    box = Box(np.array([0, 0, 0]), np.array([1, 1, 1]))
    print(box.contains(np.array([0.5, 0.5, 0.5]), 1E-13))

    sphere = Sphere(np.array([0, 0, 0]), 1)
    print(sphere.contains(np.array([0.5, 0.5, 0.5]), 1E-10))
    print(sphere.contains(np.array([1.5, 0.5, 0.5]), 1E-10))

    cyl = Cylinder(np.array([0, 0, 0]), np.array([1, 1, 1]), 1)
    print(cyl.contains(np.array([1.5, 1.5, 1.5]), 1E-10))
    print(cyl.contains(np.array([0.5, 0.5, 0.5]), 1E-10))

    cyl = Cone(np.array([0, 0, 0]), np.array([1, 1, 1]), 1, 2)
    print(cyl.contains(np.array([1.5, 1.5, 1.5]), 1E-10))
    print(cyl.contains(np.array([0.5, 0.5, 0.5]), 1E-10))

    # print(box.bbox_contains(np.array([0.5, 0.5, 0.5]), 1E-13)

    import gmsh
    import sys

    model = gmsh.model
    factory = model.occ

    gmsh.initialize(sys.argv)
    gmsh.option.setNumber("General.Terminal", 1)

    c = Cone(np.array([0, 0, 0]),
             np.array([1, 1, 2]),
             0.2,
             0.5)
    tag = c.as_gmsh(model)
    factory.synchronize();

    print(model.occ.getCenterOfMass(3, tag))
    print(c.center_of_mass)
    # model.mesh.generate(3)
    # #model.mesh.refine()
    # #model.mesh.setOrder(2)
    # #model.mesh.partition(4)
    
    # gmsh.write("boolean.msh")
    
    gmsh.finalize()
