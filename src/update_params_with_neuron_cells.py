from solver.aux import mesh_statistics
import os
from os.path import join
import yaml

results_folder = 'results'
res_files = [join(results_folder, f) for f in os.listdir(results_folder)
             if os.path.isdir(join(results_folder, f))]

for res in res_files:
    if os.path.isfile(join(res, 'params.yaml')):    
        with open(join(res, 'params.yaml'), 'r') as f:
            info = yaml.load(f)

        mesh_path = info['mesh']['path']
        if not os.path.isfile(join(res, 'params_wneuron.yaml')):
            try:
                nvertices, ncells, ncells_neuron = mesh_statistics(mesh_path)
                print mesh_path, ' FOUND!'
                info['mesh'].update({'ncells_neuron': ncells_neuron})
                print info['mesh']
                print nvertices, ncells, ncells_neuron
                with open(join(res, 'params_wneuron.yaml'), 'w') as f:
                    yaml.dump(info, f, default_flow_style=False)
            except RuntimeError:
                # print mesh_path, ' NOT FOUND!'
                pass

