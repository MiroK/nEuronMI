

def subpaths(path, size):
    # print path, size, len(path) == size, size == 2
    assert len(path) >= size >= 2

    if len(path) == size: return [path]

    if size == 2: return [[path[0], path[-1]]]

    return [[path[0]] + subpath
            for i in range(1, len(path)-size+2)
            for subpath in subpaths(path[i:], size-1)]
        
