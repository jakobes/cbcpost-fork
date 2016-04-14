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
"""
Functionality for computing connectivity of a mesh. 
"""
from dolfin import *
import numpy as np
from collections import deque

def _compute_connected_vertices(mesh, start=0):
    visited = VertexFunction("bool", mesh)
    visited.set_all(False)
    visited[start] = True
    queue = deque()
    queue.append(start)

    while len(queue) > 0:
        idx = queue.popleft()
        v = Vertex(mesh, idx)
        neighbors = set()
        N = 0
        for e in edges(v):
            neighbors.update(e.entities(0).astype(np.int))

        for j in neighbors:
            if not visited[j]:
                queue.append(j)
                visited[j] = True
    return visited



from cbcpost.utils.mpi_utils import broadcast, distribution, gather
def compute_connectivity(mesh, cell_connectivity=True):
    """Compute connected regions of mesh.
    Regions are considered connected if they share a vertex through an edge.
    """
    # Compute first vertex connectivity (defined by sharing edges)
    mesh.init(0,1)
    regions = VertexFunction("size_t", mesh)
    regions.set_all(0)
    i = 0
    
    while True:
        i += 1

        unset = np.where(regions.array()==0)[0]
        if len(unset) == 0:
            break
        start = unset[0]
        tic()
        connectivity = _compute_connected_vertices(mesh, start)
        regions.array()[connectivity.array()] = i
    
    dist = distribution(np.max(regions.array()))

    # Make all regions separate (regions on different processes are currently considered disjoint)
    regions.array()[:] += np.sum(dist[:MPI.rank(mesh.mpi_comm())])
    
    # Find global indices of all vertices shared by more than one process
    global_indices = mesh.topology().global_indices(0)
    se = mesh.topology().shared_entities(0)

    li = se.keys()
    gi_local = np.array([global_indices[i] for i in se.keys()])
    mapping = dict(zip(gi_local,li))
    gi = gather(gi_local, 0)
    gi = np.hstack(gi).flatten()
    gi = broadcast(gi, 0)
    
    # gi is now common on all processes
    
    # Connect disjointed regions through shared vertices
    while True:
        v = regions.array()[se.keys()]
        d = dict(zip(gi_local,v))
        shift = dict()
            
        for gidx in gi:
            lidx = mapping.get(gidx, -1)
            this_v = d.get(gidx, np.inf)
            v = int(MPI.min(mesh.mpi_comm(), float(this_v)))
            if this_v == v or this_v == np.inf:
                continue
            shift[this_v] = v

        # Check if no shift is needed, and if so, break
        no_shift = bool(int(MPI.min(mesh.mpi_comm(), float(int(shift==dict())))))
        if no_shift:
            break

        for k,v in shift.items():
            regions.array()[regions.array()==k] = v
        
        # Condense regions, so that M == number of regions        
        M = int(MPI.max(mesh.mpi_comm(), float(np.max(regions.array()))))
        values = np.unique(regions.array())
        for i in range(1,M+1):
            has_value = MPI.max(mesh.mpi_comm(), float(i in values))
            if has_value == 0:
                regions.array()[regions.array()>i] -= 1
                values = np.unique(regions.array())
    
    if cell_connectivity:
        cf = CellFunction("size_t", mesh)
        cf.set_all(0)
        cells = mesh.cells()[:,0]
        cf.array()[:] = regions.array()[cells]

        regions = cf

    return regions

if __name__ == '__main__':
    import mshr
    from dolfin import *
    #set_log_level(60)
    #print "hei"
    domain = mshr.Sphere(Point(-1.0,0.0,0.0), 0.8)+mshr.Sphere(Point(1.0,0.0,0.0), 0.8)
    domain += mshr.Sphere(Point(0.0,1.5,0.0), 0.8)+mshr.Sphere(Point(0.0,-1.5,0.0), 0.8)
    
    #domain = mshr.Circle(Point(-1.0,0.0), 0.8)+mshr.Circle(Point(1.0,0.0), 0.8)
    mesh = mshr.generate_mesh(domain, 20)
    #from IPython import embed; embed()
    tic()
    connectivity = compute_connectivity(mesh)
    print mesh.size_global(0), mesh.size_global(3), toc()
    
    File("connectivity.xdmf") << connectivity
    
