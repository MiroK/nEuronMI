'''
- load data from results folder
- plot minimum peak with probe, without probe, and difference depending on:
    - boxsize (1-2-3)
    - meshsize (0-1-2-3)
    - probe
'''

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
import os
from os.path import join

results = [f for f in os.listdir('results') if 'wprobe' in f]

probe = []
neuron = []
box = []
coarse = []
tip_x, tip_y, tip_z = [], [], []
min_amp_wprobe = []
min_amp_noprobe = []
diff = []
rots = []
rads = []

for res in results:
    if 'rot' in res and 'rad' in res:
        ntype, ptype, tx, ty, tz, _, cc, _, bb, _, rot,  _, rad, wp = res.split('_')
    elif 'rot' in res:
        ntype, ptype, tx, ty, tz, _, cc, _, bb, _, rot, wp = res.split('_')
        rad = 0
    else:
        ntype, ptype, tx, ty, tz, _, cc, _, bb, wp = res.split('_')
        rot = 0
        rad = 0

    folder_w = join('results', res)
    folder_no = join('results', res[:res.find('wprobe')] + 'noprobe')

    try:
        v_ext_w = np.load(join(folder_w, 'v_probe.npy'))
        v_ext_no = np.load(join(folder_no, 'v_probe.npy'))

        neuron.append(ntype)
        probe.append(ptype)
        box.append(bb)
        coarse.append(cc)
        tip_x.append(tx)
        tip_y.append(ty)
        tip_z.append(tz)
        rots.append(rot)
        rads.append(rad)
        min_amp_wprobe.append(np.min(v_ext_w))
        min_amp_noprobe.append(np.min(v_ext_no))
        diff.append(np.abs(np.min(v_ext_w) - np.min(v_ext_no)))

    except:
        pass

data = pd.DataFrame({'neuron': neuron, 'probe': probe, 'box': box, 'coarse': coarse,
                     'tip_x': tip_x, 'tip_y': tip_y, 'tip_z': tip_z, 'min_wprobe': min_amp_wprobe,
                     'min_noprobe': min_amp_noprobe, 'diff': diff, 'rot': rots, 'rad': rads})

data.to_pickle(join('results', 'results.pkl'))