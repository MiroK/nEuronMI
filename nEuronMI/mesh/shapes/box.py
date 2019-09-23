from baseshape import BaseShape
import numpy as np


class Box(BaseShape):
    '''Box is specified by min X, Y, Z coordinates and extents'''
    def __init__(self, p, dx):
        assert len(p) == 3 and len(dx) == 3
        assert np.all(dx > 0)

        self.min_ = p
        self.max_ = p + dx
        self._control_points = [self.min_, self.max_]

        self._bbox = self

    def contains(self, point, tol):
        return np.all(np.logical_and(self.min_ - tol < point,
                                     point < self.max_ + tol))

    def as_gmsh(self, factory, tag=-1):
        args = np.r_[self.min_, self.max_ - self.min_, tag]
        return factory.addBox(*args)

# --------------------------------------------------------------------

if __name__ == '__main__':

    box = Box(np.array([0, 0, 0]), np.array([1, 1, 1]))
    print box.contains(np.array([0.5, 0.5, 0.5]), 1E-13)

    print box.bbox_contains(np.array([0.5, 0.5, 0.5]), 1E-13)

    import gmsh
    import sys

    model = gmsh.model
    factory = model.occ

    gmsh.initialize(sys.argv)

    gmsh.option.setNumber("General.Terminal", 1)

    model.add("boolean")

    # from http://en.wikipedia.org/wiki/Constructive_solid_geometry

    gmsh.option.setNumber("Mesh.Algorithm", 6);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.4);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.4);

    box.as_gmsh(factory)
    factory.synchronize();

    model.mesh.generate(3)
    #model.mesh.refine()
    #model.mesh.setOrder(2)
    #model.mesh.partition(4)
    
    gmsh.write("boolean.msh")
    
    gmsh.finalize()
