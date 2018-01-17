'''
    Loads neuron geometry and plot in xy plane
'''

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

data = np.load('L5_TTPC1_cADpyr232_1_geometry.npz')

'''
parameters:

bottoms: 3d point corresponding to base of neural segment - np.array (nseg, 3)
tops: 3d point corresponding to top of neural segment - np.array (nseg, 3)
centers: 3d point corresponding center of neural segment - np.array (nseg, 3)
directions: 3d normalized vector indicating neural segment direction - np.array (nseg, 3)
lengths: length of each neural segment - np.array (nseg)
diams: diameter of each neural segment - np.array (nseg)
secs: neural structure of each segment (soma - axon - dendrite) - np.array (nseg)

'''

bottoms, tops, centers, directions, lengths, diams, secs = data['bottoms'], data['tops'], data['centers'],\
                                                           data['directions'], data['lengths'], data['diams'], \
                                                           data['secs']

# directions perpenicular to segment direction, to build polygons
perp_dir = np.array([[-dire[1], dire[0]] for dire in directions])

n=len(bottoms)
patches = []

fig = plt.figure()
ax = fig.add_subplot(111)
color = []

print bottoms[np.where(secs == 'soma')[0]]

# plot in xy plane by building polygons starting at bottom and ending at tops with different diameters
# soma is red, axon is green, dendrites are blue
for (bot, top, diam, diam_1, perp, sec) in zip(bottoms[:n-1], tops[:n-1], diams[:n-1], diams[1:n], perp_dir[:n-1],
                                               secs[:-1]):
    pol = np.squeeze(np.array([bot[:2]-perp[:2]*0.5*diam, bot[:2]+perp[:2]*0.5*diam,
                     top[:2] + perp[:2] * 0.5 * diam_1, top[:2]-perp[:2]*0.5*diam_1]))
    poly = Polygon(pol, True)
    if sec == 'soma':
        color.append('r')
    elif sec == 'axon':
        color.append('g')
    else:
        color.append('b')

    if sec == 'soma':
        patches.append(poly)

p = PatchCollection(patches, color=color, alpha=0.4)

ax.add_collection(p)
ax.axis('equal')
ax.set_xlabel('x ($\mu$m)')
ax.set_ylabel('y ($\mu$m)')
