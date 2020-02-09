Overview
========


:code:`neuronmi` is a Python package with an high-level API to simulate neurons with finite element methods.

It allows users to build meshes with neurons and recording devices, and to simulate the neuronal activity with
different models:

- 3D-3D EMI formulation

- 3D-1D EMI formulation

- *hybrid* solution (in progress)


The EMI model (Extracellular-Membrane-Intracellular) is the most advanced of these models, as it explicitly represents
the intracellular and extracellular spaces, and the neuronal membrane. This formulation enables users to simulate and
study complex phenomena, including ephaptic effects between neurons and the effect of neural devices on the recorded
signals.