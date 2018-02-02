import numpy as np
import matplotlib.pylab as plt
from neuroplot import *
from os.path import join

no_mesh = 'sphere_fancy_50.0_0.0_-150.0_noprobe'
w_mesh = 'sphere_fancy_50.0_0.0_-150.0_wprobe'

conv=1E-4

times = np.load(join('results', no_mesh, 'times.npy'))
sites = np.load(join('results', no_mesh, 'sites.npy'))

v_noprobe = np.load(join('results', no_mesh, 'v_probe.npy'))
v_wprobe = np.load(join('results', w_mesh, 'v_probe.npy'))

v_p = np.squeeze(np.array([v_noprobe, v_wprobe]))

pitch = np.array([18., 25.])*conv

if len(v_p.shape) > 2:
    fig, _, _ = plot_mea_recording(v_p, sites, pitch)
else:
    plt.plot(times, np.transpose(v_p))





