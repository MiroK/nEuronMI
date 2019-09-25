from collections import namedtuple


def as_namedtuple(d):
    '''Dictionary -> namedtuple instance'''
    keys = list(d.keys())
    foo = namedtuple('foo', keys)
    return foo(**d)


def has_positive_values(d, keys=None):
    '''Dict of positive numbers [selection of keys]'''
    if keys is None:
        return all(v > 0 for v in d.values())

    return all(d[k] > 0 for k in d if k in keys)

# --------------------------------------------------------------------

if __name__ == '__main__':
    d = {'a': 1, 'b': 2}
    print(as_namedtuple(d))
