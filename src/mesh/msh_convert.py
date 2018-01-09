import subprocess, os


def convert(msh_file, h5_file):
    '''Temporary version of convertin from msh to h5'''
    root, _ = os.path.splitext(msh_file)
    assert os.path.splitext(msh_file)[1] == '.msh'
    assert os.path.splitext(h5_file)[1] == '.h5'

    # Get the xml mesh
    xml_file = '.'.join([root, 'xml'])
    subprocess.call(['dolfin-convert %s %s' % (msh_file, xml_file)], shell=True)

    cmd = '''from dolfin import Mesh, HDF5File;\
             mesh=Mesh('%(xml_file)s');\
             assert mesh.topology().dim() == 3;\
             out=HDF5File(mesh.mpi_comm(), '%(h5_file)s', 'w');\
             out.write(mesh, 'mesh');''' % {'xml_file': xml_file,
                                             'h5_file': h5_file}

    for region in ('facet_region.xml', 'physical_region.xml'):
        name, _ = region.split('_')
        r_xml_file = '_'.join([root, region])
        if os.path.exists(r_xml_file):
            cmd_r = '''from dolfin import MeshFunction;\
                       f = MeshFunction('size_t', mesh, '%(r_xml_file)s');\
                       out.write(f, '%(name)s');\
                       ''' % {'r_xml_file': r_xml_file, 'name': name}
        
        cmd = ''.join([cmd, cmd_r])

    cmd = 'python -c "%s"' % cmd

    return subprocess.call([cmd], shell=True)


def cleanup(files=None, exts=()):
    '''Get rid of xml'''
    if files is not None:
        return map(os.remove, files)
    else:
        files = filter(lambda f: any(map(f.endswith, exts)), os.listdir('.'))
        print 'Removing', files
        return cleanup(files)
                    
# --------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import Mesh, MeshFunction, HDF5File, mpi_comm_world
    from dolfin import FacetFunction, CellFunction, File
    import argparse


    parser = argparse.ArgumentParser(description='Convert msh file to h5')
    parser.add_argument('io', type=str, nargs='+', help='input [output]')

    parser.add_argument('--save', type=int, help='save as pvd', default=0)
  
    parser.add_argument('--cleanup', type=str, nargs='+',
                        help='extensions to delete', default=())
    args = parser.parse_args()

    # Protecting self
    assert not(set(('geo', '.geo')) & set(args.cleanup))
    
    try:
        msh_file, h5_file = args.io[:2]
    except ValueError:
        msh_file = args.io[0]

        root, ext = os.path.splitext(msh_file)
        h5_file = '.'.join([root, 'h5'])

    convert(msh_file, h5_file)

    h5 = HDF5File(mpi_comm_world(), h5_file, 'r')
    mesh = Mesh()
    h5.read(mesh, 'mesh', False)

    surfaces = FacetFunction('size_t', mesh)
    h5.read(surfaces, 'facet')

    volumes = CellFunction('size_t', mesh)
    h5.read(volumes, 'physical')

    if args.save:
        File('results/%s_surf.pvd' % root) << surfaces
        File('results/%s_vols.pvd' % root) << volumes    

    cleanup(exts=args.cleanup)
