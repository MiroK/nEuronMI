'''
    Loads neuron geometry and plot in xy plane
'''

import numpy as np
import os
from os.path import join
import matplotlib.pyplot as plt
import LFPy

from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


def get_3d_cell_geometry(cell=None, cell_name=None, cell_folder=None, pos=None, rot=None):

    if cell is None:
        if cell_name is None or cell_folder is None:
            raise AttributeError('Either a Cell object or the cell name and location should be passed as parameters')
        folder = join(cell_folder, cell_name)
        cwd = os.getcwd()
        print folder
        os.chdir(folder)
        morphologyfile = os.listdir('morphology')[0]  # glob('morphology\\*')[0]
        print join('morphology', morphologyfile)
        cell = LFPy.Cell(morphology=join('morphology', morphologyfile), pt3d=True)
        os.chdir(cwd)
    elif type(cell) is not LFPy.TemplateCell and type(cell) is not LFPy.Cell:
        raise AttributeError('Either a Cell object or the cell name and location should be passed as parameters')

    # cell = return_cell_shape(folder, cell_name)
    if pos is not None:
        if len(pos) != 3:
            print 'Input a single posiion at a time'
        else:
            cell.set_pos(pos[0], pos[1], pos[2])
    if rot is not None:
        if len(rot) != 3:
            print 'Input a single posiion at a time'
        else:
            cell.set_rotation(rot[0], rot[1], rot[2])

    n3dsegments = 0
    for idx in range(len(cell.x3d)):
        n3dsegments += len(cell.x3d[idx])

    bottoms = []
    tops = []
    centers = []
    directions = []
    lengths = []
    diams = []
    secs = []

    for idx, sec in enumerate(range(len(cell.allsecnames))):
        for jj in range(len(cell.x3d[idx]) - 1):
            bottoms.append(np.array([cell.x3d[idx][jj], cell.y3d[idx][jj], cell.z3d[idx][jj]]))
            tops.append(np.array([cell.x3d[idx][jj + 1], cell.y3d[idx][jj + 1], cell.z3d[idx][jj + 1]]))
            directions.append((tops[-1] - bottoms[-1]) / np.linalg.norm(tops[-1] - bottoms[-1]))
            lengths.append(np.linalg.norm(tops[-1] - bottoms[-1]))
            centers.append(bottoms[-1]+0.5*directions[-1]*lengths[-1])
            diams.append(cell.diam3d[idx][jj])
            if 'axon' in sec:
                secs.append('axon')
            elif 'soma' in sec:
                secs.append('soma')
            elif 'dend' in sec:
                secs.append('dend')
            elif 'apic' in sec:
                secs.append('dend')

    bottoms = np.array(bottoms)
    tops = np.array(tops)
    centers = np.array(centers)
    directions = np.array(directions)
    lengths = np.array(lengths)
    diams = np.array(diams)

    return bottoms, tops, centers, directions, lengths, diams, secs


cell_folder = join(os.getcwd(), 'cell_models')
cell_names = sorted([f for f in os.listdir(cell_folder) if f.startswith('L5')])


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

bottoms, tops, centers, directions, lengths, diams, secs = get_3d_cell_geometry(cell_name=cell_names[0],
                                                                                cell_folder=cell_folder)

# directions perpenicular to segment direction, to build polygons
perp_dir = np.array([[-dire[1], dire[0]] for dire in directions])

n=len(bottoms)
patches = []

fig = plt.figure()
ax = fig.add_subplot(111)
color = []

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
    patches.append(poly)

p = PatchCollection(patches, color=color, alpha=0.4)

ax.add_collection(p)
ax.axis('equal')
ax.set_xlabel('x ($\mu$m)')
ax.set_ylabel('y ($\mu$m)')