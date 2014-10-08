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
import numpy as np  
from cbcpost.utils.mpi_utils import broadcast
from cbcpost.utils import cbc_warning
from scipy.spatial.ckdtree import cKDTree as KDTree
from dolfin import MPI, mpi_comm_world

def restriction_map(V, Vb):
    "Return a map between dofs in Vb to dofs in V. Vb's mesh should be a submesh of V's Mesh."
    
    if V.ufl_element().family() == "Discontinuous Lagrange" and V.ufl_element().degree > 0:
        raise RuntimeError("This function does not work for DG-spaces of degree >0 \
                           (several dofs associated with same point in same subspace).")
    
    if V.ufl_element().family() != "Lagrange":
        cbc_warning("This function is only tested for CG-spaces.")
    assert V.ufl_element().family() == Vb.ufl_element().family(), "ufl elements differ in the two spaces"
    assert V.ufl_element().degree() == Vb.ufl_element().degree(), "ufl elements differ in the two spaces"
    assert V.ufl_element().cell() == Vb.ufl_element().cell(), "ufl elements differ in the two spaces"
    
    
    # Recursively call this function if V has sub-spaces
    if V.num_sub_spaces() > 0:
        mapping = {}
        for i in range(V.num_sub_spaces()):
            mapping.update(restriction_map(V.sub(i), Vb.sub(i)))

        return mapping
    
    D = V.mesh().geometry().dim()
    
    dm = V.dofmap()
    dmb = Vb.dofmap()
    
    N = len(dm.dofs())
    Nb = len(dmb.dofs())
    
    dofs = dm.dofs()
    
    # Extract coordinates of dofs
    if dm.is_view():
        coords = V.collapse().dofmap().tabulate_all_coordinates(V.mesh()).reshape(N, D)
        coordsb = Vb.collapse().dofmap().tabulate_all_coordinates(Vb.mesh()).reshape(Nb,D)
    else:
        coords = V.dofmap().tabulate_all_coordinates(V.mesh()).reshape(N, D)
        coordsb = Vb.dofmap().tabulate_all_coordinates(Vb.mesh()).reshape(Nb,D)
    
    # Build KDTree to compute distances from coordinates in base
    kdtree = KDTree(coords)
    
    eps = 1e-12
    
    mapping = {}
    request_dofs = np.array([])

    for i, subdof in enumerate(dmb.dofs()):
        # Find closest dof in base
        d, idx = kdtree.query(coordsb[i])
        if d < eps:
            # Dof found on this process, add to map
            dof = dofs[idx]
            assert subdof not in mapping
            mapping[subdof] = dof
        else:
            # Search for this dof on other processes
            add_dofs = np.hstack(([subdof], coordsb[i]))
            request_dofs = np.append(request_dofs, add_dofs)

    # Scatter all dofs not found on current process to all processes
    all_request_dofs = [None]*MPI.size(mpi_comm_world())
    for j in xrange(MPI.size(mpi_comm_world())):
        all_request_dofs[j] = broadcast(request_dofs, j)
    
    # Re-order all requested dofs
    # Remove items coming from this process
    all_request_dofs[MPI.rank(mpi_comm_world())] = []
    all_request_dofs = np.hstack(all_request_dofs)
    
    all_request_dofs = all_request_dofs.reshape(len(all_request_dofs)/(D+1), D+1)
    all_request_dofs = dict(zip(all_request_dofs[:,0], all_request_dofs[:,1:]))

    # Search this process for all dofs not found on same process as subdof
    for subdof, coordsbi in all_request_dofs.items():
        subdof = int(subdof)
        
        # Find closest dof in base
        d, idx = kdtree.query(coordsbi)
        if d < eps:
            # Dof found on this process, add to map
            dof = dofs[idx]
            assert subdof not in mapping
            mapping[subdof] = dof

    return mapping
        
if __name__ == '__main__':
    from dolfin import (UnitCubeMesh, SubDomain, MeshFunction, tic, toc, VectorFunctionSpace,
                        FunctionSpace, Function, assemble, Expression, project, dx, inner)
    N = 4
    mesh = UnitCubeMesh(N,N,N)
    
    class Left(SubDomain):
        def inside(self, x, on_boundary):
            return x[0] > 0.7
    
    from cbcpost.utils import create_submesh
    
    
    markers = MeshFunction("size_t", mesh, 3)
    markers.set_all(0)
    Left().mark(markers, 1)
    tic()
    mesh2 = create_submesh(mesh, markers, 1)
    print "Time create submesh: ", toc()
    #bmesh.coordinates()[:] += 0.1
    #bmesh2 = Mesh("submesh.xml")
    
    #print bmesh2.size_global(0)
    #print bmesh2.size_global(2)
    V = FunctionSpace(mesh, "CG", 1)
    Vb = FunctionSpace(mesh2, "CG", 1)
    
    tic()
    mapping = restriction_map(V, Vb)
    print "Time restriction_map: ", toc()
    
    expr = Expression("x[0]*x[1]+x[2]*x[2]+3.0")
    u = project(expr, V)
    u2 = Function(Vb)
    u2.vector()[mapping.keys()] = u.vector()[mapping.values()]
    
    print assemble(u*dx(1, subdomain_data=markers)), assemble(u2*dx)
    
    V = VectorFunctionSpace(mesh, "CG", 1)
    Vb = VectorFunctionSpace(mesh2, "CG", 1)
    
    mapping = restriction_map(V, Vb)
    
    expr = Expression(("x[0]*x[1]+x[2]*x[2]+3.0", "2+x[1]*x[2]", "x[0]+3*x[2]"))
    u = project(expr, V)
    u2 = Function(Vb)
    u2.vector()[mapping.keys()] = u.vector()[mapping.values()]
    
    print assemble(inner(u,u)*dx(1, subdomain_data=markers)), assemble(inner(u2,u2)*dx)