from libc.math cimport fmin, fmax, sqrt

cimport numpy as np
import numpy as np


cdef distance2d(double x1, double y1, double x2, double y2):
    cdef double d, dx, dy
    dx = (x1 - x2)
    dy = (y1 - y2)
    d = sqrt(dx*dx + dy*dy)
    return d


cdef distance3d(double x1, double y1, double z1, double x2, double y2, double z2):
    cdef double d, dx, dy, dz
    dx = (x1 - x2)
    dy = (y1 - y2)
    dz = (z1 - z2)
    d = sqrt(dx*dx + dy*dy + dz*dz)
    return d


cdef double c2d( np.ndarray[np.float64_t, ndim=2] ca, int i, int j,
                 np.ndarray[np.float64_t ,ndim=2] P,
                 np.ndarray[np.float64_t, ndim=2] Q):
    if ca[i, j] > -1:
        return ca[i,j]
    elif i == 0 and j == 0:
        ca[i, j] = distance2d(P[0, 0], P[0, 1], Q[0, 0], Q[0, 1])
    elif i > 0 and j == 0:
        ca[i, j] = fmax(c2d(ca, i-1, 0, P, Q),
                        distance2d(P[i, 0], P[i, 1], Q[0, 0], Q[0, 1]))
    elif i == 0 and j > 0:
        ca[i, j] = fmax(c2d(ca, 0, j-1, P, Q),
                        distance2d(P[0, 0], P[0, 1], Q[j, 0], Q[j, 1]))
    elif i > 0 and j > 0:
        ca[i, j] = fmax(fmin(c2d(ca, i-1, j, P, Q),
                             fmin(c2d(ca, i-1, j-1, P, Q),
                                  c2d(ca, i, j-1, P, Q))),
                        distance2d(P[i, 0], P[i, 1], Q[j, 0], Q[j, 1]))
    else:
        ca[i, j] = float("inf")
        
    return ca[i, j]


cdef double c3d( np.ndarray[np.float64_t, ndim=2] ca, int i, int j,
                 np.ndarray[np.float64_t ,ndim=3] P,
                 np.ndarray[np.float64_t, ndim=3] Q):
    if ca[i, j] > -1:
        return ca[i,j]
    elif i == 0 and j == 0:
        ca[i, j] = distance3d(P[0, 0], P[0, 1], P[0, 2], Q[0, 0], Q[0, 1], Q[0, 2])
    elif i > 0 and j == 0:
        ca[i, j] = fmax(c3d(ca, i-1, 0, P, Q),
                        distance3d(P[i, 0], P[i, 1], P[i, 2], Q[0, 0], Q[0, 1], Q[0, 2]))
    elif i == 0 and j > 0:
        ca[i, j] = fmax(c3d(ca, 0, j-1, P, Q),
                        distance3d(P[0, 0], P[0, 1], P[0, 2], Q[j, 0], Q[j, 1], Q[j, 2]))
    elif i > 0 and j > 0:
        ca[i, j] = fmax(fmin(c3d(ca, i-1, j, P, Q),
                             fmin(c3d(ca, i-1, j-1, P, Q),
                                  c3d(ca, i, j-1, P, Q))),
                        distance3d(P[i, 0], P[i, 1], P[i, 2], Q[j, 0], Q[j, 1], Q[j, 2]))
    else:
        ca[i, j] = float("inf")
        
    return ca[i, j]


def frechet_distance(P, Q):
    '''
    Discrete Frecht distance of curves P and Q. A curve is an ordered
    set of vertices.

    Following "Computing discrete Frechet distance" by Eiter&Mannila
    '''
    _, dimP = P.shape
    _, dimQ = Q.shape

    assert dimP == dimQ

    ca = -1.0*np.ones((len(P), len(Q)))
    
    if dimP == 2:
        return c2d(ca, len(P) - 1, len(Q) - 1, P, Q)
    else:
        return c3d(ca, len(P) - 1, len(Q) - 1, P, Q)
