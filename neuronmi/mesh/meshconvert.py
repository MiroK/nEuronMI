""" Module for converting various mesh formats."""

# Copyright (C) 2006 Anders Logg
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
#
# Modified by Garth N. Wells (gmsh function)
# Modified by Alexander H. Jarosch (gmsh fix)
# Modified by Angelo Simone (Gmsh and Medit fix)
# Modified by Andy R. Terrel (gmsh fix and triangle function)
# Modified by Magnus Vikstrom (metis and scotch function)
# Modified by Bartosz Sawicki (diffpack function)
# Modified by Gideon Simpson (Exodus II function)
# Modified by Kent-Andre Mardal (Star-CD function)
# Modified by Nuno Lopes (fix for emc2 mesh format (medit version 0))
# Modified by Neilen Marais (add gmsh support for reading physical region)
# Modified by Evan Lezar (add support for reading gmsh physical regions on facets)
# Modified by Jan Blechta (add triangle support for marker on edges and attributes on triangles)
#
# Last changed: 2014-02-06

# NOTE: This module does not depend on (py)dolfin beeing installed.
# NOTE: If future additions need that please import dolfin in a try: except:
# NOTE: clause and tell the user to install dolfin if it is not installed.

import getopt
import sys
import re
import warnings
import os.path

from dolfin_utils.meshconvert import abaqus
from dolfin_utils.meshconvert import xml_writer
import numpy


def format_from_suffix(suffix):
    "Return format for given suffix"
    if suffix == "xml":
        return "xml"
    elif suffix == "mesh":
        return "mesh"
    elif suffix == "gmsh":
        return "gmsh"
    elif suffix == "msh":
        return "gmsh"
    elif suffix == "gra":
        return "metis"
    elif suffix == "grf":
        return "scotch"
    elif suffix == "grid":
        return "diffpack"
    elif suffix == "inp":
        return "abaqus"
    elif suffix == "ncdf":
        return "NetCDF"
    elif suffix == "exo":
        return "ExodusII"
    elif suffix == "e":
        return "ExodusII"
    elif suffix == "vrt" or suffix == "cel":
        return "StarCD"
    elif suffix == "ele" or suffix == "node":
        return "Triangle"
    else:
        _error("Sorry, unknown suffix %s." % suffix)


def mesh2xml(ifilename, ofilename):
    """Convert between .mesh and .xml, parser implemented as a
    state machine:

        0 = read 'Dimension'
        1 = read dimension
        2 = read 'Vertices'
        3 = read number of vertices
        4 = read next vertex
        5 = read 'Triangles' or 'Tetrahedra'
        6 = read number of cells
        7 = read next cell
        8 = done

    """

    print("Converting from Medit format (.mesh) to DOLFIN XML format")

    # Open files
    ifile = open(ifilename, "r")
    ofile = open(ofilename, "w")

    # Scan file for cell type
    cell_type = None
    dim = 0
    while 1:

        # Read next line
        line = ifile.readline()
        if not line: break

        # Remove newline
        if line[-1] == "\n":
            line = line[:-1]

        # Read dimension
        if line == "Dimension" or line == " Dimension":
            line = ifile.readline()
            num_dims = int(line)
            if num_dims == 2:
                cell_type = "triangle"
                dim = 2
            elif num_dims == 3:
                cell_type = "tetrahedron"
                dim = 3
            break

    # Check that we got the cell type
    if cell_type == None:
        _error("Unable to find cell type.")

    # Step to beginning of file
    ifile.seek(0)

    # Write header
    xml_writer.write_header_mesh(ofile, cell_type, dim)

    # Current state
    state = 0

    # Write data
    num_vertices_read = 0
    num_cells_read = 0

    while 1:

        # Read next line
        line = ifile.readline()
        if not line: break

        # Skip comments
        if line[0] == '#':
            continue

        # Remove newline
        if line[-1] == "\n":
            line = line[:-1]

        if state == 0:
            if line == "Dimension" or line == " Dimension":
                state += 1
        elif state == 1:
            num_dims = int(line)
            state += 1
        elif state == 2:
            if line == "Vertices" or line == " Vertices":
                state += 1
        elif state == 3:
            num_vertices = int(line)
            xml_writer.write_header_vertices(ofile, num_vertices)
            state += 1
        elif state == 4:
            if num_dims == 2:
                (x, y, tmp) = line.split()
                x = float(x)
                y = float(y)
                z = 0.0
            elif num_dims == 3:
                (x, y, z, tmp) = line.split()
                x = float(x)
                y = float(y)
                z = float(z)
            xml_writer.write_vertex(ofile, num_vertices_read, x, y, z)
            num_vertices_read += 1
            if num_vertices == num_vertices_read:
                xml_writer.write_footer_vertices(ofile)
                state += 1
        elif state == 5:
            if (line == "Triangles" or line == " Triangles") and num_dims == 2:
                state += 1
            if line == "Tetrahedra" and num_dims == 3:
                state += 1
        elif state == 6:
            num_cells = int(line)
            xml_writer.write_header_cells(ofile, num_cells)
            state += 1
        elif state == 7:
            if num_dims == 2:
                (n0, n1, n2, tmp) = line.split()
                n0 = int(n0) - 1
                n1 = int(n1) - 1
                n2 = int(n2) - 1
                xml_writer.write_cell_triangle(ofile, num_cells_read, n0, n1, n2)
            elif num_dims == 3:
                (n0, n1, n2, n3, tmp) = line.split()
                n0 = int(n0) - 1
                n1 = int(n1) - 1
                n2 = int(n2) - 1
                n3 = int(n3) - 1
                xml_writer.write_cell_tetrahedron(ofile, num_cells_read, n0, n1, n2, n3)
            num_cells_read += 1
            if num_cells == num_cells_read:
                xml_writer.write_footer_cells(ofile)
                state += 1
        elif state == 8:
            break

    # Check that we got all data
    if state == 8:
        print("Conversion done")
    else:
        _error("Missing data, unable to convert")

    # Write footer
    xml_writer.write_footer_mesh(ofile)

    # Close files
    ifile.close()
    ofile.close()


def gmsh2xml(ifilename, handler):
    """Convert between .gmsh v2.0 format (http://www.geuz.org/gmsh/) and .xml,
    parser implemented as a state machine:

        0 = read 'MeshFormat'
        1 = read  mesh format data
        2 = read 'EndMeshFormat'
        3 = read 'Nodes'
        4 = read  number of vertices
        5 = read  vertices
        6 = read 'EndNodes'
        7 = read 'Elements'
        8 = read  number of cells
        9 = read  cells
        10 = done

    Afterwards, extract physical region numbers if they are defined in
    the mesh file as a mesh function.

    """

    print("Converting from Gmsh format (.msh, .gmsh) to DOLFIN XML format")

    # The dimension of the gmsh element types supported here as well as the dolfin cell types for each dimension
    gmsh_dim = {15: 0, 1: 1, 2: 2, 4: 3}
    cell_type_for_dim = {1: "interval", 2: "triangle", 3: "tetrahedron"}
    # the gmsh element types supported for conversion
    supported_gmsh_element_types = [1, 2, 4, 15]

    # Open files
    ifile = open(ifilename, "r")

    # Scan file for cell type
    cell_type = None
    highest_dim = 0
    line = ifile.readline()
    while line:

        # Remove newline
        if line[-1] == "\n":
            line = line[:-1]

        # Read dimension
        if line.find("$Elements") == 0:

            line = ifile.readline()
            num_elements = int(line)
            if num_elements == 0:
                _error("No elements found in gmsh file.")
            line = ifile.readline()

            # Now iterate through elements to find largest dimension.  Gmsh
            # format might include elements of lower dimensions in the element list.
            # We also need to count number of elements of correct dimensions.
            # Also determine which vertices are not used.
            dim_count = {0: 0, 1: 0, 2: 0, 3: 0}
            vertices_used_for_dim = {0: [], 1: [], 2: [], 3: []}
            # Array used to store gmsh tags for 1D (type 1/line), 2D (type 2/triangular) elements and 3D (type 4/tet) elements
            tags_for_dim = {0: [], 1: [], 2: [], 3: []}

            while line.find("$EndElements") == -1:
                element = line.split()
                elem_type = int(element[1])
                num_tags = int(element[2])
                if elem_type in supported_gmsh_element_types:
                    dim = gmsh_dim[elem_type]
                    if highest_dim < dim:
                        highest_dim = dim
                    node_num_list = [int(node) for node in element[3 + num_tags:]]
                    vertices_used_for_dim[dim].extend(node_num_list)
                    if num_tags > 0:
                        tags_for_dim[dim].append(tuple(int(tag) for tag in element[3:3 + num_tags]))
                    dim_count[dim] += 1
                else:
                    # TODO: output a warning here. "gmsh element type %d not supported" % elem_type
                    pass
                line = ifile.readline()
        else:
            # Read next line
            line = ifile.readline()

    # Check that we got the cell type and set num_cells_counted
    if highest_dim == 0:
        _error("Unable to find cells of supported type.")

    num_cells_counted = dim_count[highest_dim]
    vertex_set = set(vertices_used_for_dim[highest_dim])
    vertices_used_for_dim[highest_dim] = None

    vertex_dict = {}
    for n, v in enumerate(vertex_set):
        vertex_dict[v] = n

    # Step to beginning of file
    ifile.seek(0)

    # Set mesh type
    handler.set_mesh_type(cell_type_for_dim[highest_dim], highest_dim)

    # Initialise node list (gmsh does not export all vertexes in order)
    nodelist = {}

    # Current state
    state = 0

    # Write data
    num_vertices_read = 0
    num_cells_read = 0

    # Only import the dolfin objects if facet markings exist
    process_facets = False
    if any(len(tags_for_dim[dim]) > 0 for dim in (highest_dim - 1, 1)):
        # first construct the mesh
        try:
            from dolfin import MeshEditor, Mesh
        except ImportError:
            _error("DOLFIN must be installed to handle Gmsh boundary regions")
        mesh = Mesh()
        mesh_editor = MeshEditor()
        cell_type = {1: 'interval', 2: 'triangle', 3: 'tetrahedron'}[highest_dim]
        mesh_editor.open(mesh, cell_type, highest_dim, highest_dim)
        process_facets = True
    else:
        # TODO: Output a warning or an error here
        me = None

    while state != 10:

        # Read next line
        line = ifile.readline()
        if not line: break

        # Skip comments
        if line[0] == '#':
            continue

        # Remove newline
        if line[-1] == "\n":
            line = line[:-1]

        if state == 0:
            if line == "$MeshFormat":
                state = 1
        elif state == 1:
            (version, file_type, data_size) = line.split()
            state = 2
        elif state == 2:
            if line == "$EndMeshFormat":
                state = 3
        elif state == 3:
            if line == "$Nodes":
                state = 4
        elif state == 4:
            num_vertices = len(vertex_dict)
            handler.start_vertices(num_vertices)
            if process_facets:
                mesh_editor.init_vertices_global(num_vertices, num_vertices)
            state = 5
        elif state == 5:
            (node_no, x, y, z) = line.split()
            node_no = int(node_no)
            x, y, z = [float(xx) for xx in (x, y, z)]
            if node_no in vertex_dict:
                node_no = vertex_dict[node_no]
            else:
                continue
            nodelist[int(node_no)] = num_vertices_read
            handler.add_vertex(num_vertices_read, [x, y, z])
            if process_facets:
                if highest_dim == 1:
                    coords = numpy.array([x])
                elif highest_dim == 2:
                    coords = numpy.array([x, y])
                elif highest_dim == 3:
                    coords = numpy.array([x, y, z])
                mesh_editor.add_vertex(num_vertices_read, coords)

            num_vertices_read += 1

            if num_vertices == num_vertices_read:
                handler.end_vertices()
                state = 6
        elif state == 6:
            if line == "$EndNodes":
                state = 7
        elif state == 7:
            if line == "$Elements":
                state = 8
        elif state == 8:
            handler.start_cells(num_cells_counted)
            if process_facets:
                mesh_editor.init_cells_global(num_cells_counted, num_cells_counted)

            state = 9
        elif state == 9:
            element = line.split()
            elem_type = int(element[1])
            num_tags = int(element[2])
            if elem_type in supported_gmsh_element_types:
                dim = gmsh_dim[elem_type]
            else:
                dim = 0
            if dim == highest_dim:
                node_num_list = [vertex_dict[int(node)] for node in element[3 + num_tags:]]
                for node in node_num_list:
                    if not node in nodelist:
                        _error("Vertex %d of %s %d not previously defined." %
                               (node, cell_type_for_dim[dim], num_cells_read))
                cell_nodes = [nodelist[n] for n in node_num_list]
                handler.add_cell(num_cells_read, cell_nodes)

                if process_facets:
                    cell_nodes = numpy.array([nodelist[n] for n in node_num_list], dtype=numpy.uintp)
                    mesh_editor.add_cell(num_cells_read, cell_nodes)

                num_cells_read += 1

            if num_cells_counted == num_cells_read:
                handler.end_cells()
                if process_facets:
                    mesh_editor.close()
                state = 10
        elif state == 10:
            break

    # Write mesh function based on the Physical Regions defined by
    # gmsh, but only if they are not all zero. All zero physical
    # regions indicate that no physical regions were defined.
    if highest_dim not in [1, 2, 3]:
        _error("Gmsh tags not supported for dimension %i. Probably a bug" % dim)

    tags = tags_for_dim[highest_dim]
    physical_regions = tuple(tag[0] for tag in tags)
    if not all(tag == 0 for tag in physical_regions):
        handler.start_meshfunction("volume_region", dim, num_cells_counted)
        for i, physical_region in enumerate(physical_regions):
            handler.add_entity_meshfunction(i, physical_region)
        handler.end_meshfunction()

    # Now process the facet markers
    tags = tags_for_dim[highest_dim - 1]
    if (len(tags) > 0) and (mesh is not None):
        physical_regions = tuple(tag[0] for tag in tags)
        if not all(tag == 0 for tag in physical_regions):
            mesh.init(highest_dim - 1, 0)

            # Get the facet-node connectivity information (reshape as a row of node indices per facet)
            if highest_dim == 1:
                # for 1d meshes the mesh topology returns the vertex to vertex map, which isn't what we want
                # as facets are vertices
                facets_as_nodes = numpy.array([[i] for i in range(mesh.num_facets())])
            else:
                facets_as_nodes = numpy.array(mesh.topology()(highest_dim - 1, 0)(),
                                              dtype='uintp').reshape(mesh.num_facets(), highest_dim)

            # Build the reverse map
            nodes_as_facets = {}
            for facet in range(mesh.num_facets()):
                nodes_as_facets[tuple(facets_as_nodes[facet, :])] = facet

            data = [int(0 * k) for k in range(mesh.num_facets())]
            for i, physical_region in enumerate(physical_regions):
                nodes = [n - 1 for n in
                         vertices_used_for_dim[highest_dim - 1][highest_dim * i:(highest_dim * i + highest_dim)]]
                nodes.sort()

                if physical_region != 0:
                    try:
                        index = nodes_as_facets[tuple(nodes)]
                        data[index] = physical_region
                    except IndexError:
                        raise Exception("The facet (%d) was not found to mark: %s" % (i, nodes))

            # Create and initialise the mesh function
            handler.start_meshfunction("facet_region", highest_dim - 1, mesh.num_facets())
            for index, physical_region in enumerate(data):
                handler.add_entity_meshfunction(index, physical_region)
            handler.end_meshfunction()

    # Edge markers
    if highest_dim == 3:
        tags = tags_for_dim[1]
        if (len(tags) > 0) and (mesh is not None):
            physical_regions = tuple(tag[0] for tag in tags)
            if not all(tag == 0 for tag in physical_regions):
                mesh.init(1, 0)

                # Get the edge-node connectivity information (reshape as a row
                # of node indices per edge)
                edges_as_nodes = \
                    mesh.topology()(1, 0)().reshape(mesh.num_edges(), 2)

                # Build the reverse map
                nodes_as_edges = {}
                for edge in range(mesh.num_edges()):
                    nodes_as_edges[tuple(edges_as_nodes[edge])] = edge

                data = numpy.zeros(mesh.num_edges())
                for i, physical_region in enumerate(physical_regions):
                    nodes = [n - 1 for n in vertices_used_for_dim[1][2 * i:(2 * i + 2)]]
                    nodes.sort()

                    if physical_region != 0:
                        try:
                            index = nodes_as_edges[tuple(nodes)]
                            data[index] = physical_region
                        except IndexError:
                            raise Exception("The edge (%d) was not found to mark: %s" % (i, nodes))

                # Create and initialise the mesh function
                handler.start_meshfunction("curve_region", 1, mesh.num_edges())
                for index, physical_region in enumerate(data):
                    handler.add_entity_meshfunction(index, physical_region)
                handler.end_meshfunction()

    # Check that we got all data
    if state == 10:
        print("Conversion done")
    else:
        _error("Missing data, unable to convert \n\ Did you use version 2.0 of the gmsh file format?")

    # Close files
    ifile.close()


def triangle2xml(ifilename, ofilename):
    """Convert between triangle format
    (http://www.cs.cmu.edu/~quake/triangle.html) and .xml.  The
    given ifilename should be the prefix for the corresponding
    .node, and .ele files.
    """

    def get_next_line(fp):
        """Helper function for skipping comments and blank lines"""
        line = fp.readline()
        if line == '':
            _error("Hit end of file prematurely.")
        line = line.strip()
        if not (line.startswith('#') or line == ''):
            return line
        return get_next_line(fp)

    print("Converting from Triangle format {.node, .ele} to DOLFIN XML format")

    # Open files
    for suffix in [".node", ".ele"]:
        if suffix in ifilename and ifilename[-len(suffix):] == suffix:
            ifilename = ifilename.replace(suffix, "")
    node_file = open(ifilename + ".node", "r")
    ele_file = open(ifilename + ".ele", "r")
    ofile = open(ofilename, "w")
    try:
        edge_file = open(ifilename + ".edge", "r")
        print("Found .edge file")
    except IOError:
        edge_file = None

    # Read all the nodes
    nodes = {}
    num_nodes, dim, attr, bound = map(int, get_next_line(node_file).split())
    while len(nodes) < num_nodes:
        node, x, y = get_next_line(node_file).split()[:3]
        nodes[int(node)] = (float(x), float(y))

    # Read all the triangles
    tris = {}
    tri_attrs = {}
    num_tris, n_per_tri, attrs = map(int, get_next_line(ele_file).split())
    while len(tris) < num_tris:
        line = get_next_line(ele_file).split()
        tri, n1, n2, n3 = map(int, line[:4])
        # vertices are ordered according to current UFC ordering scheme -
        # - may change in future!
        tris[tri] = tuple(sorted((n1, n2, n3)))
        tri_attrs[tri] = tuple(map(float, line[4:4 + attrs]))

    # Read all the boundary markers from edges
    edge_markers_global = {}
    edge_markers_local = []
    got_negative_edge_markers = False
    if edge_file is not None:
        num_edges, num_edge_markers = map(int, get_next_line(edge_file).split())
        if num_edge_markers == 1:
            while len(edge_markers_global) < num_edges:
                edge, v1, v2, marker = map(int, get_next_line(edge_file).split())
                if marker < 0: got_negative_edge_markers = True
                edge_markers_global[tuple(sorted((v1, v2)))] = marker
            if got_negative_edge_markers:
                print("Some edge markers are negative! dolfin will increase " \
                      "them by probably 2**32 when loading xml. " \
                      "Consider using non-negative edge markers only.")
            for tri, vertices in tris.iteritems():
                v0, v1, v2 = sorted((vertices[0:3]))
                try:
                    edge_markers_local.append((tri, 0, \
                                               edge_markers_global[(v1, v2)]))
                    edge_markers_local.append((tri, 1, \
                                               edge_markers_global[(v0, v2)]))
                    edge_markers_local.append((tri, 2, \
                                               edge_markers_global[(v0, v1)]))
                except IndexError:
                    raise Exception("meshconvert.py: The facet was not found.")
        elif num_edge_markers == 0:
            print("...but no markers in it. Ignoring it")
        else:
            print("...but %d markers specified in it. It won't be processed." \
                  % num_edge_markers)

    # Write everything out
    xml_writer.write_header_mesh(ofile, "triangle", 2)
    xml_writer.write_header_vertices(ofile, num_nodes)
    node_off = 0 if nodes.has_key(0) else -1
    for node, node_t in nodes.iteritems():
        xml_writer.write_vertex(ofile, node + node_off, node_t[0], node_t[1], 0.0)
    xml_writer.write_footer_vertices(ofile)
    xml_writer.write_header_cells(ofile, num_tris)
    tri_off = 0 if tris.has_key(0) else -1
    for tri, tri_t in tris.iteritems():
        xml_writer.write_cell_triangle(ofile, tri + tri_off, tri_t[0] + node_off,
                                       tri_t[1] + node_off, tri_t[2] + node_off)
    xml_writer.write_footer_cells(ofile)
    if len(edge_markers_local) > 0:
        xml_writer.write_header_domains(ofile)
        xml_writer.write_header_meshvaluecollection(ofile, \
                                                    "edge markers", 1, len(edge_markers_local), "uint")
        for tri, local_edge, marker in edge_markers_local:
            xml_writer.write_entity_meshvaluecollection(ofile, \
                                                        1, tri + tri_off, marker, local_edge)
        xml_writer.write_footer_meshvaluecollection(ofile)
        xml_writer.write_footer_domains(ofile)
    xml_writer.write_footer_mesh(ofile)
    for i in range(attrs):
        afilename = ofilename.replace(".xml", ".attr" + str(i) + ".xml")
        afile = open(afilename, "w")
        xml_writer.write_header_meshfunction2(afile)
        xml_writer.write_header_meshvaluecollection(afile, \
                                                    "triangle attribs " + str(i), 2, num_tris, "double")
        for tri, tri_a in tri_attrs.iteritems():
            xml_writer.write_entity_meshvaluecollection(afile, \
                                                        2, tri + tri_off, tri_a[i], 0)
        xml_writer.write_footer_meshvaluecollection(afile)
        xml_writer.write_footer_meshfunction(afile)
        print("triangle attributes from .ele file written to " + afilename)
        afile.close()

    # Close files
    node_file.close()
    ele_file.close()
    if edge_file is not None:
        edge_file.close()
    ofile.close()


def xml_old2xml(ifilename, ofilename):
    "Convert from old DOLFIN XML format to new."

    print("Converting from old (pre DOLFIN 0.6.2) to new DOLFIN XML format...")

    # Open files
    ifile = open(ifilename, "r")
    ofile = open(ofilename, "w")

    # Scan file for cell type (assuming there is just one)
    cell_type = None
    dim = 0
    while 1:

        # Read next line
        line = ifile.readline()
        if not line: break

        # Read dimension
        if "<triangle" in line:
            cell_type = "triangle"
            dim = 2
            break
        elif "<tetrahedron" in line:
            cell_type = "tetrahedron"
            dim = 3
            break

    # Step to beginning of file
    ifile.seek(0)

    # Read lines and make changes
    while 1:

        # Read next line
        line = ifile.readline()
        if not line: break

        # Modify line
        if "xmlns" in line:
            line = "<dolfin xmlns:dolfin=\"http://fenicsproject.org\">\n"
        if "<mesh>" in line:
            line = "  <mesh celltype=\"%s\" dim=\"%d\">\n" % (cell_type, dim)
        if dim == 2 and " z=\"0.0\"" in line:
            line = line.replace(" z=\"0.0\"", "")
        if " name=" in line:
            line = line.replace(" name=", " index=")
        if " name =" in line:
            line = line.replace(" name =", " index=")
        if "n0" in line:
            line = line.replace("n0", "v0")
        if "n1" in line:
            line = line.replace("n1", "v1")
        if "n2" in line:
            line = line.replace("n2", "v2")
        if "n3" in line:
            line = line.replace("n3", "v3")

        # Write line
        ofile.write(line)

    # Close files
    ifile.close();
    ofile.close();
    print("Conversion done")


def metis_graph2graph_xml(ifilename, ofilename):
    "Convert from Metis graph format to DOLFIN Graph XML."

    print("Converting from Metis graph format to DOLFIN Graph XML.")

    # Open files
    ifile = open(ifilename, "r")
    ofile = open(ofilename, "w")

    # Read number of vertices and edges
    line = ifile.readline()
    if not line:
        _error("Empty file")

    (num_vertices, num_edges) = line.split()

    xml_writer.write_header_graph(ofile, "directed")
    xml_writer.write_header_vertices(ofile, int(num_vertices))

    for i in range(int(num_vertices)):
        line = ifile.readline()
        edges = line.split()
        xml_writer.write_graph_vertex(ofile, i, len(edges))

    xml_writer.write_footer_vertices(ofile)
    xml_writer.write_header_edges(ofile, 2 * int(num_edges))

    # Step to beginning of file and skip header info
    ifile.seek(0)
    ifile.readline()
    for i in range(int(num_vertices)):
        print("vertex %g", i)
        line = ifile.readline()
        edges = line.split()
        for e in edges:
            xml_writer.write_graph_edge(ofile, i, int(e))

    xml_writer.write_footer_edges(ofile)
    xml_writer.write_footer_graph(ofile)

    # Close files
    ifile.close();
    ofile.close();


def scotch_graph2graph_xml(ifilename, ofilename):
    "Convert from Scotch graph format to DOLFIN Graph XML."

    print("Converting from Scotch graph format to DOLFIN Graph XML.")

    # Open files
    ifile = open(ifilename, "r")
    ofile = open(ofilename, "w")

    # Skip graph file version number
    ifile.readline()

    # Read number of vertices and edges
    line = ifile.readline()
    if not line:
        _error("Empty file")

    (num_vertices, num_edges) = line.split()

    # Read start index and numeric flag
    # Start index is 0 or 1 (C/Fortran)
    # Numeric flag is 3 bits where bit 1 enables vertex labels
    # bit 2 enables edge weights and bit 3 enables vertex weights

    line = ifile.readline()
    (start_index, numeric_flag) = line.split()

    # Handling not implented
    if not numeric_flag == "000":
        _error("Handling of scotch vertex labels, edge- and vertex weights not implemented")

    xml_writer.write_header_graph(ofile, "undirected")
    xml_writer.write_header_vertices(ofile, int(num_vertices))

    # Read vertices and edges, first number gives number of edges from this vertex (not used)
    for i in range(int(num_vertices)):
        line = ifile.readline()
        edges = line.split()
        xml_writer.write_graph_vertex(ofile, i, len(edges) - 1)

    xml_writer.write_footer_vertices(ofile)
    xml_writer.write_header_edges(ofile, int(num_edges))

    # Step to beginning of file and skip header info
    ifile.seek(0)
    ifile.readline()
    ifile.readline()
    ifile.readline()
    for i in range(int(num_vertices)):
        line = ifile.readline()

        edges = line.split()
        for j in range(1, len(edges)):
            xml_writer.write_graph_edge(ofile, i, int(edges[j]))

    xml_writer.write_footer_edges(ofile)
    xml_writer.write_footer_graph(ofile)

    # Close files
    ifile.close();
    ofile.close();


def diffpack2xml(ifilename, ofilename):
    "Convert from Diffpack tetrahedral/triangle grid format to DOLFIN XML."

    print(diffpack2xml.__doc__)

    # Format strings for MeshFunction XML files
    meshfunction_header = """\
<?xml version="1.0" encoding="UTF-8"?>\n
<dolfin xmlns:dolfin="http://www.fenics.org/dolfin/">
  <mesh_function type="uint" dim="%d" size="%d">\n"""
    meshfunction_entity = "    <entity index=\"%d\" value=\"%d\"/>\n"
    meshfunction_footer = "  </mesh_function>\n</dolfin>"

    # Open files
    ifile = open(ifilename, "r")
    ofile = open(ofilename, "w")

    # Read and analyze header
    while 1:
        line = ifile.readline()
        if not line:
            _error("Empty file")
        if line[0] == "#":
            break
        if re.search(r"Number of elements", line):
            num_cells = int(re.match(r".*\s(\d+).*", line).group(1))
        if re.search(r"Number of nodes", line):
            num_vertices = int(re.match(r".*\s(\d+).*", line).group(1))
        if re.search(r"Number of space dim.", line):
            num_dims = int(re.match(r".*\s(\d+).*", line).group(1))

    if num_dims == 3:
        xml_writer.write_header_mesh(ofile, "tetrahedron", 3)
        elem_type = "ElmT4n3D"
        write_cell_func = xml_writer.write_cell_tetrahedron
    else:
        xml_writer.write_header_mesh(ofile, "triangle", 2)
        elem_type = "ElmT3n2D"
        write_cell_func = xml_writer.write_cell_triangle

    xml_writer.write_header_vertices(ofile, num_vertices)

    # Read & write vertices and collect markers for vertices
    vertex_markers = []
    unique_vertex_markers = set()
    for i in range(num_vertices):
        line = ifile.readline()
        m = re.match(r"^.*\(\s*(.*)\s*\).*\](.*)$", line)
        x = map(float, re.split("[\s,]+", m.group(1)))
        xml_writer.write_vertex(ofile, i, *x)
        markers = map(int, m.group(2).split())
        vertex_markers.append(markers)
        unique_vertex_markers.update(markers)

    xml_writer.write_footer_vertices(ofile)
    xml_writer.write_header_cells(ofile, num_cells)

    # Output unique vertex markers as individual VertexFunctions
    unique_vertex_markers.difference_update([0])
    for unique_marker in unique_vertex_markers:
        ofile_marker = open(ofilename.replace(".xml", "") + \
                            "_marker_" + str(unique_marker) + ".xml", "w")
        xml_writer.write_header_meshfunction(ofile_marker, 0, num_vertices)
        for ind, markers in enumerate(vertex_markers):
            if unique_marker in markers:
                xml_writer.write_entity_meshfunction(ofile_marker, ind, unique_marker)
            else:
                xml_writer.write_entity_meshfunction(ofile_marker, ind, 0)
        xml_writer.write_footer_meshfunction(ofile_marker)

    # Ignore comment lines
    while 1:
        line = ifile.readline()
        if not line:
            _error("Empty file")
        if line[0] == "#":
            break

    # Read & write cells and collect cell and face markers
    cell_markers = []
    facet_markers = []
    facet_to_vert = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]
    vert_to_facet = facet_to_vert  # The same!

    cell_ind = 0
    while cell_ind < num_cells:
        line = ifile.readline()
        v = line.split()
        if not v:
            continue

        if v[1] != elem_type:
            _error("Only tetrahedral (ElmT4n3D) and triangular (ElmT3n2D) elements are implemented.")

        # Store Cell markers
        cell_markers.append(int(v[2]))

        # Sort vertex indices
        cell_indices = sorted(map(lambda x: int(x) - 1, v[3:]))
        write_cell_func(ofile, cell_ind, *cell_indices)

        if num_dims == 2:
            cell_ind += 1
            continue

        # Check Facet info
        process_facet = set(range(4))
        for local_vert_ind, global_vert_ind in enumerate(cell_indices):

            # If no marker is included for vertex skip corresponding facet
            if not vertex_markers[global_vert_ind]:
                process_facet.difference_update(facet_to_vert[local_vert_ind])

        # Process facets
        for local_facet in process_facet:

            # Start with markers from first vertex
            global_first_vertex = cell_indices[facet_to_vert[local_facet][0]]
            marker_intersection = set(vertex_markers[global_first_vertex])

            # Process the other vertices
            for local_vert in facet_to_vert[local_facet][1:]:
                marker_intersection.intersection_update( \
                    vertex_markers[cell_indices[local_vert]])

                if not marker_intersection:
                    break

            # If not break we have a marker on local_facet
            else:
                assert (len(marker_intersection) == 1)
                facet_markers.append((cell_ind, local_facet, \
                                      marker_intersection.pop()))
        # Bump cell_ind
        cell_ind += 1

    xml_writer.write_footer_cells(ofile)
    xml_writer.write_header_domains(ofile)

    # Write facet markers if any
    if facet_markers:
        xml_writer.write_header_meshvaluecollection(ofile, "m", 2, \
                                                    len(facet_markers), "uint")
        for cell, local_facet, marker in facet_markers:
            xml_writer.write_entity_meshvaluecollection(ofile, 2, cell, \
                                                        marker, local_facet)
        xml_writer.write_footer_meshvaluecollection(ofile)

    xml_writer.write_header_meshvaluecollection(ofile, "m", num_dims, \
                                                len(cell_markers), "uint")
    for cell, marker in enumerate(cell_markers):
        xml_writer.write_entity_meshvaluecollection(ofile, num_dims, cell, \
                                                    marker)
    xml_writer.write_footer_meshvaluecollection(ofile)
    xml_writer.write_footer_domains(ofile)
    xml_writer.write_footer_mesh(ofile)

    # Close files
    ifile.close()
    ofile.close()


class ParseError(Exception):
    """ Error encountered in source file.
    """


class DataHandler(object):
    """ Baseclass for handlers of mesh data.

    The actual handling of mesh data encountered in the source file is
    delegated to a polymorfic object. Typically, the delegate will write the
    data to XML.
    @ivar _state: the state which the handler is in, one of State_*.
    @ivar _cell_type: cell type in mesh. One of CellType_*.
    @ivar _dim: mesh dimensions.
    """
    State_Invalid, State_Init, State_Vertices, State_Cells, \
    State_MeshFunction, State_MeshValueCollection = range(6)
    CellType_Tetrahedron, CellType_Triangle, CellType_Interval = range(3)

    def __init__(self):
        self._state = self.State_Invalid

    def set_mesh_type(self, cell_type, dim):
        assert self._state == self.State_Invalid
        self._state = self.State_Init
        if cell_type == "tetrahedron":
            self._cell_type = self.CellType_Tetrahedron
        elif cell_type == "triangle":
            self._cell_type = self.CellType_Triangle
        elif cell_type == "interval":
            self._cell_type = self.CellType_Interval
        self._dim = dim

    def start_vertices(self, num_vertices):
        assert self._state == self.State_Init
        self._state = self.State_Vertices

    def add_vertex(self, vertex, coords):
        assert self._state == self.State_Vertices

    def end_vertices(self):
        assert self._state == self.State_Vertices
        self._state = self.State_Init

    def start_cells(self, num_cells):
        assert self._state == self.State_Init
        self._state = self.State_Cells

    def add_cell(self, cell, nodes):
        assert self._state == self.State_Cells

    def end_cells(self):
        assert self._state == self.State_Cells
        self._state = self.State_Init

    def start_domains(self):
        assert self._state == self.State_Init

    def end_domains(self):
        self._state = self.State_Init

    def start_meshfunction(self, name, dim, size):
        assert self._state == self.State_Init
        self._state = self.State_MeshFunction

    def add_entity_meshfunction(self, index, value):
        assert self._state == self.State_MeshFunction

    def end_meshfunction(self):
        assert self._state == self.State_MeshFunction
        self._state = self.State_Init

    def start_mesh_value_collection(self, name, dim, size, etype):
        assert self._state == self.State_Init
        self._state = self.State_MeshValueCollection

    def add_entity_mesh_value_collection(self, dim, index, value, local_entity=0):
        assert self._state == self.State_MeshValueCollection

    def end_mesh_value_collection(self):
        assert self._state == self.State_MeshValueCollection
        self._state = self.State_Init

    def warn(self, msg):
        """ Issue warning during parse.
        """
        warnings.warn(msg)

    def error(self, msg):
        """ Raise error during parse.

        This method is expected to raise ParseError.
        """
        raise ParseError(msg)

    def close(self):
        self._state = self.State_Invalid


class XmlHandler(DataHandler):
    """ Data handler class which writes to Dolfin XML.
    """

    def __init__(self, ofilename):
        DataHandler.__init__(self)
        self._ofilename = ofilename
        self.__ofile = open(ofilename, "w")
        self.__ofile_meshfunc = None

    def ofile(self):
        return self.__ofile

    def set_mesh_type(self, cell_type, dim):
        DataHandler.set_mesh_type(self, cell_type, dim)
        xml_writer.write_header_mesh(self.__ofile, cell_type, dim)

    def start_vertices(self, num_vertices):
        DataHandler.start_vertices(self, num_vertices)
        xml_writer.write_header_vertices(self.__ofile, num_vertices)

    def add_vertex(self, vertex, coords):
        DataHandler.add_vertex(self, vertex, coords)
        xml_writer.write_vertex(self.__ofile, vertex, *coords)

    def end_vertices(self):
        DataHandler.end_vertices(self)
        xml_writer.write_footer_vertices(self.__ofile)

    def start_cells(self, num_cells):
        DataHandler.start_cells(self, num_cells)
        xml_writer.write_header_cells(self.__ofile, num_cells)

    def add_cell(self, cell, nodes):
        DataHandler.add_cell(self, cell, nodes)
        if self._cell_type == self.CellType_Tetrahedron:
            func = xml_writer.write_cell_tetrahedron
        elif self._cell_type == self.CellType_Triangle:
            func = xml_writer.write_cell_triangle
        elif self._cell_type == self.CellType_Interval:
            func = xml_writer.write_cell_interval

        func(self.__ofile, cell, *nodes)

    def end_cells(self):
        DataHandler.end_cells(self)
        xml_writer.write_footer_cells(self.__ofile)

    def start_meshfunction(self, name, dim, size):
        DataHandler.start_meshfunction(self, name, dim, size)
        fname = os.path.splitext(self.__ofile.name)[0]
        self.__ofile_meshfunc = open("%s_%s.xml" % (fname, name), "w")
        xml_writer.write_header_meshfunction(self.__ofile_meshfunc, dim, size)

    def add_entity_meshfunction(self, index, value):
        DataHandler.add_entity_meshfunction(self, index, value)
        xml_writer.write_entity_meshfunction(self.__ofile_meshfunc, index, value)

    def end_meshfunction(self):
        DataHandler.end_meshfunction(self)
        xml_writer.write_footer_meshfunction(self.__ofile_meshfunc)
        self.__ofile_meshfunc.close()
        self.__ofile_meshfunc = None

    def start_domains(self):
        # DataHandler.start_domains(self)
        xml_writer.write_header_domains(self.__ofile)

    def end_domains(self):
        # DataHandler.end_domains(self)
        xml_writer.write_footer_domains(self.__ofile)

    def start_mesh_value_collection(self, name, dim, size, etype):
        DataHandler.start_mesh_value_collection(self, name, dim, size, etype)
        xml_writer.write_header_meshvaluecollection(self.__ofile, name, dim, size, etype)

    def add_entity_mesh_value_collection(self, dim, index, value, local_entity=0):
        DataHandler.add_entity_mesh_value_collection(self, dim, index, value)
        xml_writer.write_entity_meshvaluecollection(self.__ofile, dim, index, value, local_entity=local_entity)

    def end_mesh_value_collection(self):
        DataHandler.end_mesh_value_collection(self)
        xml_writer.write_footer_meshvaluecollection(self.__ofile)

    def close(self):
        DataHandler.close(self)
        if self.__ofile.closed:
            return
        xml_writer.write_footer_mesh(self.__ofile)
        self.__ofile.close()
        if self.__ofile_meshfunc is not None:
            self.__ofile_meshfunc.close()


def netcdf2xml(ifilename, ofilename):
    "Convert from NetCDF format to DOLFIN XML."

    print("Converting from NetCDF format (.ncdf) to DOLFIN XML format")

    # Open files
    ifile = open(ifilename, "r")
    ofile = open(ofilename, "w")

    cell_type = None
    dim = 0

    # Scan file for dimension, number of nodes, number of elements
    while 1:
        line = ifile.readline()
        if not line:
            _error("Empty file")
        if re.search(r"num_dim.*=", line):
            dim = int(re.match(".*\s=\s(\d+)\s;", line).group(1))
        if re.search(r"num_nodes.*=", line):
            num_vertices = int(re.match(".*\s=\s(\d+)\s;", line).group(1))
        if re.search(r"num_elem.*=", line):
            num_cells = int(re.match(".*\s=\s(\d+)\s;", line).group(1))
        if re.search(r"connect1 =", line):
            break

    num_dims = dim

    # Set cell type
    if dim == 2:
        cell_type = "triangle"
    if dim == 3:
        cell_type = "tetrahedron"

    # Check that we got the cell type
    if cell_type == None:
        _error("Unable to find cell type.")

    # Write header
    xml_writer.write_header_mesh(ofile, cell_type, dim)
    xml_writer.write_header_cells(ofile, num_cells)
    num_cells_read = 0

    # Read and write cells
    while 1:
        # Read next line
        line = ifile.readline()
        if not line:
            break
        connect = re.split("[,;]", line)
        if num_dims == 2:
            n0 = int(connect[0]) - 1
            n1 = int(connect[1]) - 1
            n2 = int(connect[2]) - 1
            xml_writer.write_cell_triangle(ofile, num_cells_read, n0, n1, n2)
        elif num_dims == 3:
            n0 = int(connect[0]) - 1
            n1 = int(connect[1]) - 1
            n2 = int(connect[2]) - 1
            n3 = int(connect[3]) - 1
            xml_writer.write_cell_tetrahedron(ofile, num_cells_read, n0, n1, n2, n3)
        num_cells_read += 1
        if num_cells == num_cells_read:
            xml_writer.write_footer_cells(ofile)
            xml_writer.write_header_vertices(ofile, num_vertices)
            break

    num_vertices_read = 0
    coords = [[], [], []]
    coord = -1

    while 1:
        line = ifile.readline()
        if not line:
            _error("Missing data")
        if re.search(r"coord =", line):
            break

    # Read vertices
    while 1:
        line = ifile.readline()
        if not line:
            break
        if re.search(r"\A\s\s\S+,", line):
            coord += 1
            print("Found x_" + str(coord) + " coordinates")
        coords[coord] += line.split()
        if re.search(r";", line):
            break

    # Write vertices
    for i in range(num_vertices):
        if num_dims == 2:
            x = float(re.split(",", coords[0].pop(0))[0])
            y = float(re.split(",", coords[1].pop(0))[0])
            z = 0
        if num_dims == 3:
            x = float(re.split(",", coords[0].pop(0))[0])
            y = float(re.split(",", coords[1].pop(0))[0])
            z = float(re.split(",", coords[2].pop(0))[0])
        xml_writer.write_vertex(ofile, i, x, y, z)

    # Write footer
    xml_writer.write_footer_vertices(ofile)
    xml_writer.write_footer_mesh(ofile)

    # Close files
    ifile.close()
    ofile.close()


def _error(message):
    "Write an error message"
    for line in message.split("\n"):
        print("*** %s" % line)
    sys.exit(2)


def convert2xml(ifilename, ofilename, iformat=None):
    """ Convert a file to the DOLFIN XML format.
    """
    convert(ifilename, XmlHandler(ofilename), iformat=iformat)


def convert(ifilename, handler, iformat=None):
    """ Convert a file using a provided data handler.

    Note that handler.close is called when this function finishes.
    @param ifilename: Name of input file.
    @param handler: The data handler (instance of L{DataHandler}).
    @param iformat: Format of input file.
    """
    if iformat is None:
        iformat = format_from_suffix(os.path.splitext(ifilename)[1][1:])
    # XXX: Backwards-compat
    if hasattr(handler, "_ofilename"):
        ofilename = handler._ofilename
    # Choose conversion
    if iformat == "mesh":
        # Convert from mesh to xml format
        mesh2xml(ifilename, ofilename)
    elif iformat == "gmsh":
        # Convert from gmsh to xml format
        gmsh2xml(ifilename, handler)
    elif iformat == "Triangle":
        # Convert from Triangle to xml format
        triangle2xml(ifilename, ofilename)
    elif iformat == "xml-old":
        # Convert from old to new xml format
        xml_old2xml(ifilename, ofilename)
    elif iformat == "metis":
        # Convert from metis graph to dolfin graph xml format
        metis_graph2graph_xml(ifilename, ofilename)
    elif iformat == "scotch":
        # Convert from scotch graph to dolfin graph xml format
        scotch_graph2graph_xml(ifilename, ofilename)
    elif iformat == "diffpack":
        # Convert from Diffpack tetrahedral grid format to xml format
        diffpack2xml(ifilename, ofilename)
    elif iformat == "abaqus":
        # Convert from abaqus to xml format
        abaqus.convert(ifilename, handler)
    elif iformat == "NetCDF":
        # Convert from NetCDF generated from ExodusII format to xml format
        netcdf2xml(ifilename, ofilename)
    elif iformat == "ExodusII":
        # Convert from ExodusII format to xml format via NetCDF
        exodus2xml(ifilename, ofilename)
    elif iformat == "StarCD":
        # Convert from Star-CD tetrahedral grid format to xml format
        starcd2xml(ifilename, ofilename)
    else:
        _error("Sorry, cannot convert between %s and DOLFIN xml file formats." % iformat)

    # XXX: handler.close messes things for other input formats than abaqus or gmsh
    if iformat in ("abaqus", "gmsh"):
        handler.close()


def starcd2xml(ifilename, ofilename):
    "Convert from Star-CD tetrahedral grid format to DOLFIN XML."

    print(starcd2xml.__doc__)

    if not os.path.isfile(ifilename[:-3] + "vrt") or not os.path.isfile(ifilename[:-3] + "cel"):
        print("StarCD format requires one .vrt file and one .cel file")
        sys.exit(2)

    # open output file
    ofile = open(ofilename, "w")

    # Open file, the vertices are in a .vrt file
    ifile = open(ifilename[:-3] + "vrt", "r")

    write_header_mesh(ofile, "tetrahedron", 3)

    # Read & write vertices

    # first, read all lines (need to sweep to times through the file)
    lines = ifile.readlines()

    # second, find the number of vertices
    num_vertices = -1
    counter = 0
    # nodenr_map is needed because starcd support node numbering like 1,2,4 (ie 3 is missing)
    nodenr_map = {}
    for line in lines:
        nodenr = int(line[0:15])
        nodenr_map[nodenr] = counter
        counter += 1
    num_vertices = counter

    # third, run over all vertices
    xml_writer.write_header_vertices(ofile, num_vertices)
    for line in lines:
        nodenr = int(line[0:15])
        vertex0 = float(line[15:31])
        vertex1 = float(line[31:47])
        vertex2 = float(line[47:63])
        xml_writer.write_vertex(ofile, nodenr_map[nodenr], float(vertex0), float(vertex1), float(vertex2))
    xml_writer.write_footer_vertices(ofile)

    # Open file, the cells are in a .cel file
    ifile = open(ifilename[:-3] + "cel", "r")

    # Read & write cells

    # first, read all lines (need to sweep to times through the file)
    lines = ifile.readlines()

    # second, find the number of cells
    num_cells = -1
    counter = 0
    for line in lines:
        l = [int(a) for a in line.split()]
        cellnr, node0, node1, node2, node3, node4, node5, node6, node7, tmp1, tmp2 = l
        if node4 > 0:
            if node2 == node3 and node4 == node5 and node5 == node6 and node6 == node7:  # these nodes should be equal
                counter += 1
            else:
                print("The file does contain cells that are not tetraheders. The cell number is ", cellnr,
                      " the line read was ", line)
        else:
            # triangles on the surface
            #            print("The file does contain cells that are not tetraheders node4==0. The cell number is ", cellnr, " the line read was ", line
            # sys.exit(2)
            pass

    num_cells = counter

    # third, run over all cells
    xml_writer.write_header_cells(ofile, num_cells)
    counter = 0
    for line in lines:
        l = [int(a) for a in line.split()]
        cellnr, node0, node1, node2, node3, node4, node5, node6, node7, tmp1, tmp2 = l
        if (node4 > 0):
            if node2 == node3 and node4 == node5 and node5 == node6 and node6 == node7:  # these nodes should be equal

                xml_writer.write_cell_tetrahedron(ofile, counter, nodenr_map[node0], nodenr_map[node1],
                                                  nodenr_map[node2], nodenr_map[node4])
                counter += 1

    xml_writer.write_footer_cells(ofile)
    xml_writer.write_footer_mesh(ofile)

    # Close files
    ifile.close()
    ofile.close()
