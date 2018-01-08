import numpy as np


# -------------------------------------------------------------------

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from math import sqrt, sin, cos, pi
    import random

    # FIXME: True random data! Random in all the vars
    npoints = 100

    u0, v0 = 1., 2.
    
    au = 1.
    av = 0.4

    b0 = 0.2
    b1 = 0.4

    c = 0.2

    # ------

    # ------
    
    u = np.array([u0, v0]); u /= np.linalg.norm(u)
    v = np.array([-v0, u0]); v /= np.linalg.norm(v)

    xs, ys = [], []
    for aui in np.random.normal(au, au/20, npoints):
        for avi in np.random.normal(av, av/20, npoints):
            t = 2*pi*random.random()
            x, y = aui*cos(t), avi*sin(t)
            xs.append(x)
            ys.append(y)

    plt.figure()
    plt.plot(xs, ys, marker='x', linestyle='none')
    plt.show()
    
    def fun(x, pts):
        x = u0, v0, a, b, c
