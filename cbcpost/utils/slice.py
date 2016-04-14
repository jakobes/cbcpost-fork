# Copyright (C) 2010-2014 Simula Research Laboratory
#
# This file is part of CBCPOST.
#
# CBCPOST is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CBCPOST is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with CBCPOST. If not, see <http://www.gnu.org/licenses/>.
"""Functionality for slicing a mesh."""
from dolfin import Constant, MeshEditor, Mesh, MPI, mpi_comm_world
from cbcpost.utils.mpi_utils import distribute_meshdata, distribution
from cbcpost.utils.connectivity import compute_connectivity
from cbcpost.utils.submesh import create_submesh
import numpy as np

class Slice(Mesh):
    "Deprecated Slice-class"
    def __init__(self, *args, **kwargs):
        raise RuntimeError("Deprecated. Use create_slice to create slicemesh.")

def create_slice(basemesh, point, normal, closest_region=False, crinkle_clip=False):
    """Create a slicemesh from a basemesh.

    :param basemesh: Mesh to slice
    :param point: Point in slicing plane
    :param normal: Normal to slicing plane
    :param closest_region: Set to True to extract disjoint region closest to specified point
    :param crinkle_clip: Set to True to return mesh of same topological dimension as basemesh

    .. note::

        Only 3D-meshes currently supported for slicing.

    .. warning::

        Slice-instances are intended for visualization only, and may produce erronous
        results if used for computations.

    """
    assert basemesh.geometry().dim() == 3, "Can only slice 3D-meshes."

    P = np.array([point[0], point[1], point[2]])

    # Create unit normal
    n = np.array([normal[0],normal[1], normal[2]])
    n = n/np.linalg.norm(n)
    #self.n = Constant((n[0], n[1], n[2]))

    # Calculate the distribution of vertices around the plane
    # (sign of np.dot(p-P, n) determines which side of the plane p is on)
    vsplit = np.dot(basemesh.coordinates()-P, n)

    # Count each cells number of vertices on the "positive" side of the plane
    # Only cells with vertices on both sides of the plane intersect the plane
    operator = np.less
    npos = np.sum(vsplit[basemesh.cells()] < 0, 1)
    
    intersection_cells = basemesh.cells()[(npos > 0) & (npos < 4)]

    if len(intersection_cells) == 0:
        # Try to put "zeros" on other side of plane
        # FIXME: handle cells with vertices exactly intersecting the plane in a more robust manner.
        operator = np.greater
        npos = np.sum(vsplit[basemesh.cells()] > 0, 1)
        #cell_indices = (npos > 0) & (npos < 4)
        intersection_cells = basemesh.cells()[(npos > 0) & (npos < 4)]

    if crinkle_clip:
        cf = CellFunction("size_t", basemesh)
        cf.set_all(0)
        cf.array()[(npos>0) & (npos<4)] = 1
        mesh = create_submesh(basemesh, cf, 1)
    else:
        def add_cell(cells, cell):
            # Split cell into triangles
            for i in xrange(len(cell)-2):
                cells.append(cell[i:i+3])
    
        cells = []
        index = 0
        indexes = {}
        for c in intersection_cells:
            a = operator(vsplit[c], 0)
            positives = c[np.where(a==True)[0]]
            negatives = c[np.where(a==False)[0]]
    
            cell = []
            for pp_ind in positives:
                pp = basemesh.coordinates()[pp_ind]
    
                for pn_ind in negatives:
                    pn = basemesh.coordinates()[pn_ind]
                    if (pp_ind, pn_ind) not in indexes:
                        # Calculate intersection point with the plane
                        d = np.dot(P-pp, n)/np.dot(pp-pn, n)
                        ip = pp+(pp-pn)*d
    
                        indexes[(pp_ind, pn_ind)] = (index, ip)
                        index += 1
    
                    cell.append(indexes[(pp_ind, pn_ind)][0])
    
    
            add_cell(cells, cell)
        MPI.barrier(mpi_comm_world())
    
        # Assign global indices
        # TODO: Assign global indices properly
        dist = distribution(index)
        global_idx = sum(dist[:MPI.rank(mpi_comm_world())])
        vertices = {}
        for idx, p in indexes.values():
            vertices[idx] = (global_idx, p)
            global_idx += 1
    
    
        global_num_cells = MPI.sum(mpi_comm_world(), len(cells))
        global_num_vertices = MPI.sum(mpi_comm_world(), len(vertices))
        
        mesh = Mesh()
    
        # Return empty mesh if no intersections were found
        if global_num_cells == 0:
            mesh_editor = MeshEditor()
            mesh_editor.open(mesh, "triangle", 2, 3)
    
            mesh_editor.init_vertices(0)
            mesh_editor.init_cells(0)
    
            mesh_editor.close()
        else:

            # Distribute mesh if empty on any processors
            cells, vertices = distribute_meshdata(cells, vertices)
        
            # Build mesh
            mesh_editor = MeshEditor()
            mesh_editor.open(mesh, "triangle", 2, 3)
        
            mesh_editor.init_vertices(len(vertices))
            mesh_editor.init_cells(len(cells))
        
            for index, cell in enumerate(cells):
                mesh_editor.add_cell(index, cell[0], cell[1], cell[2])
        
            for local_index, (global_index, coordinates) in vertices.items():
                mesh_editor.add_vertex_global(int(local_index), int(global_index), coordinates)
        
            mesh_editor.close()
            mesh.topology().init(0, len(vertices), global_num_vertices)
            mesh.topology().init(2, len(cells), global_num_cells)

    
    if closest_region and mesh.size_global(0) > 0:
        assert MPI.size(mpi_comm_world())==1, "Extract closest region does not work in parallel"
        regions = compute_connectivity(mesh)
        i,d = mesh.bounding_box_tree().compute_closest_entity(Point(P))
        
        if d == MPI.min(mesh.mpi_comm(), d):
            v = regions[int(i)]
        else:
            v = 0
        
        v = MPI.max(mesh.mpi_comm(), v)
        mesh = create_submesh(mesh, regions, v)

    return mesh


if __name__ == '__main__':
    import mshr
    from dolfin import *
    set_log_level(100)
    domain = mshr.Sphere(Point(-1.0,0.0,0.0), 1.2)+mshr.Sphere(Point(1.0,0.0,0.0), 1.2)
    mesh = mshr.generate_mesh(domain, 30)
    p = np.array([1.0,1.0,0.0])
    
    n = np.array([0,1,0])

    slicemesh = create_slice(mesh, p, n, closest_region=False, crinkle_clip=True)

    plot(slicemesh)
    interactive()

    from dolfin import File

    File("basemesh.pvd") << mesh
    File("slice_mesh.xdmf") << slicemesh
