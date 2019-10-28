from neuronmi.mesh.shapes.utils import first, second
import json


class EMIEntityMap(object):
    '''
    In EMI model we tags surfaces(2) and volumes(3). For each we then
    have a mapping of shape name to named surfaces/volumes with their
    geometrical and physical tags
    '''
    def __init__(self, tagged_volumes=None, tagged_surfaces=None, json_fp=None):
        # JSON load
        if json_fp is not None:
            d = json.load(json_fp)
            tagged_volumes, tagged_surfaces = d['3'], d['2']
        
        self.volumes = tagged_volumes
        self.surfaces = tagged_surfaces

        self._nn = sum(1 for k in self.volumes.keys() if 'neuron_' in k)
        assert self._nn == sum(1 for k in self.surfaces.keys() if 'neuron_' in k)

    @property
    def num_neurons(self):
        return self._nn
    
    def dump(self, fp):
        '''JSON dump'''
        return json.dump({'3': self.volumes, '2': self.surfaces}, fp)

    def volume_entity_tags(self, shape):
        '''Entity tags of `shape`'s volumes'''
        return self.entity_tags(self.volumes, first, shape)
    
    def volume_physical_tags(self, shape):
        '''Physical tags of `shape`'s volumes'''
        return self.entity_tags(self.volumes, second, shape)

    def surface_entity_tags(self, shape):
        '''Entity tags of `shape`'s surfaces'''
        return self.entity_tags(self.surfaces, first, shape)

    def surface_physical_tags(self, shape):
        '''Physical tags of `shape`'s surfaces'''
        return self.entity_tags(self.surfaces, second, shape)

    def entity_tags(self, entities, access, shape):
        '''Work horse of getting tags'''
        if shape is 'all_neurons':
            result = {}
            for i in range(self.num_neurons):
                neuron = 'neuron_%d' % i
                result.update({'_'.join([neuron, k]): v
                               for k, v in self.entity_tags(entities, access, neuron).items()})
            return result
        
        if isinstance(shape, int):
            shape = 'neuron_%d' % shape
            
        return {k: access(v) for k, v in entities[shape].items()}

