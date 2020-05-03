import MEAutility as mu
import numpy as np
import matplotlib.pylab as plt

eap_folder = 'probe_effect/neuronexus/'

u_with = np.load(eap_folder + 'u_with.npy').T * 1000
u_without = np.load(eap_folder + 'u_without.npy').T * 1000
centers = np.load(eap_folder + 'centers.npy')

probe = mu.return_mea(info={'pos': centers})

vscale = np.max(np.abs(u_with))

ax = mu.plot_mea_recording(u_without, probe, vscale=vscale, colors='C0', lw=1.5)
ax = mu.plot_mea_recording(u_with, probe, vscale=vscale, colors='C1', ax=ax, lw=1.5)

plt.ion()
plt.show()