import numpy as np
import pandas as pd
import yaml
import os
from os.path import join

res_folder = 'results'

files = [f for f in os.listdir(res_folder)]

coarse = []
boxsize = []
cells = []
vertices = []
facets = []
system = []
times = []

for f in files:
    if 'noprobe' in f:
        with open(join(res_folder, f, 'params.yaml'), 'r') as file:
            info = yaml.load(file)

        coarse.append(int(f[f.find('coarse_')+7:f.find('coarse_') + 7 + 1]))
        boxsize.append(int(f[f.find('box_') + 4:f.find('box_') + 4 + 1]))
        cells.append(int(info['mesh']['cells']))
        vertices.append(int(info['mesh']['vertices']))
        facets.append(int(info['mesh']['facets']))
        system.append(int(info['performance']['system size']))
        times.append(round(float(info['performance']['time']),2))

order = np.argsort(coarse)[::-1]
coarse = np.array(coarse)
boxsize = np.array(boxsize)
cells = np.array(cells)
vertices = np.array(vertices)
facets = np.array(facets)
system = np.array(system)
times = np.array(times)

data = {'Coarse': coarse[order], 'Box size': boxsize[order], 'Cells': cells[order],
        'Vertices': vertices[order], 'Facets': facets[order], 'System size': system[order], 'T (s)': times[order]}

df = pd.DataFrame(data)

print  df.to_latex(columns=['Coarse', 'Box size', 'Cells', 'Vertices', 'Facets', 'System size', 'T (s)'], index=False)