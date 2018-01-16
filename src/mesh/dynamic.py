def memoize(f):
    '''Memoize a faction of one hashable arguments'''
    memory = {}
    def helper(arg):
        if arg not in memory: memory[arg] = f(arg)

        return memory[arg]
    return helper

@memoize
def nsubpaths((path, size)):
    '''Computes the number of paths of size formed from given path'''
    if path == size: return 1
    if size == 2: return 1

    return sum(nsubpaths((path-i, size-1)) for i in range(1, path-size+2))



# import matplotlib.pyplot as plt

# path = 100
# sizes = range(2, path+1)
# lengths = [nsubpaths((path, size)) for size in sizes]

# plt.figure()
# plt.semilogy(sizes, lengths, 'bx--')
# plt.show()
