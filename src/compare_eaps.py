import numpy as np
import matplotlib.pylab as plt

try:
    from neuroplot import *
    neur_plot=True
except ImportError:
    neur_plot=False
from os.path import join
import sys


if __name__ == '__main__':

    if '-f' in sys.argv:
        pos = sys.argv.index('-f')
        mesh = sys.argv[pos + 1]
    else:
        raise Exception('Indicate fem results with -f argument')

    if 'noprobe' in mesh:
        no_mesh = mesh
        w_mesh = mesh[:-8] + 'wprobe'
    elif 'wprobe' in mesh:
        w_mesh = mesh
        no_mesh = mesh[:-7] + 'noprobe'

    print w_mesh
    print no_mesh

    conv=1E-4

    times = np.load(join(no_mesh, 'times.npy'))
    sites = np.load(join(no_mesh, 'sites.npy'))

    v_noprobe = np.load(join(no_mesh, 'v_probe.npy'))
    v_wprobe = np.load(join(w_mesh, 'v_probe.npy'))

    v_p = np.squeeze(np.array([v_noprobe, v_wprobe]))

    pitch = np.array([18., 25.])*conv

    if len(v_p.shape) > 2:
	if neur_plot:
            fig, _, _ = plot_mea_recording(v_p, sites, pitch)
    else:
        plt.plot(times, np.transpose(v_p))

    print 'NO PROBE: ', np.min(v_noprobe)
    print 'WITH PROBE: ', np.min(v_wprobe)
    print 'DIFF: ', np.min(v_noprobe) - np.min(v_wprobe)

    plt.ion()
    plt.show()
