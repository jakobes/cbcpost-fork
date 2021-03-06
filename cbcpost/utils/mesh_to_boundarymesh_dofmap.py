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

from dolfin import Cell, Facet
from numpy import where
from distutils.version import LooseVersion, StrictVersion

def boundarymesh_to_mesh_dofmap(boundary, Vb, V):
    "Find the mapping from dofs on boundary FS to dofs on full mesh FS"
    mapping = mesh_to_boundarymesh_dofmap(boundary, V, Vb, _should_own="bdof")
    mapping = dict((v,k) for k,v in mapping.iteritems())
    return mapping

def mesh_to_boundarymesh_dofmap(boundary, V, Vb, _should_own="cdof"):
    "Find the mapping from dofs on full mesh FS to dofs on boundarymesh FS"
    from dolfin import dolfin_version, MPI, mpi_comm_world
    #if dolfin_version() != '1.4.0' and MPI.size(mpi_comm_world()) > 1:
    #    raise RuntimeError("mesh_to_boundarymesh_dofmap is currently not supported in parallel in version %s" %(dolfin_version()))

    assert V.ufl_element().family() == Vb.ufl_element().family()
    assert V.ufl_element().degree() == Vb.ufl_element().degree()
    assert _should_own in ["cdof", "bdof"]

    # Currently only CG1 and DG0 spaces are supported
    assert V.ufl_element().family() in ["Lagrange", "Discontinuous Lagrange"]
    if V.ufl_element().family() == "Discontinuous Lagrange":
        assert V.ufl_element().degree() == 0
    else:
        assert V.ufl_element().degree() == 1

    D = boundary.topology().dim()
    mesh = V.mesh()

    V_dm = V.dofmap()
    Vb_dm = Vb.dofmap()

    dofmap_to_boundary = {}

    # Extract maps from boundary to mesh
    vertex_map = boundary.entity_map(0)
    cell_map = boundary.entity_map(D)

    for i in xrange(len(cell_map)):
        boundary_cell = Cell(boundary, i)
        mesh_facet = Facet(mesh, cell_map[i])
        mesh_cell_index = mesh_facet.entities(D+1)[0]
        mesh_cell = Cell(mesh, mesh_cell_index)

        cell_dofs = V_dm.cell_dofs(mesh_cell_index)
        boundary_dofs = Vb_dm.cell_dofs(i)

        if V_dm.num_entity_dofs(0) > 0:
            for v_idx in boundary_cell.entities(0):

                mesh_v_idx = vertex_map[int(v_idx)]
                mesh_list_idx = where(mesh_cell.entities(0) == mesh_v_idx)[0][0]
                boundary_list_idx = where(boundary_cell.entities(0) == v_idx)[0][0]

                bdofs = boundary_dofs[Vb_dm.tabulate_entity_dofs(0, boundary_list_idx)]
                cdofs = cell_dofs[V_dm.tabulate_entity_dofs(0, mesh_list_idx)]

                for bdof, cdof in zip(bdofs, cdofs):
                    #if dolfin_version() in ["1.4.0+", "1.5.0", "1.6.0"]:
                    if LooseVersion(dolfin_version()) > LooseVersion("1.4.0"):
                        bdof = Vb_dm.local_to_global_index(bdof)
                        cdof = V_dm.local_to_global_index(cdof)

                    if _should_own == "cdof" and not (V_dm.ownership_range()[0] <= cdof < V_dm.ownership_range()[1]):
                        continue
                    elif _should_own == "bdof" and not (Vb_dm.ownership_range()[0] <= bdof < Vb_dm.ownership_range()[1]):
                        continue
                    else:
                        dofmap_to_boundary[bdof] = cdof

        if V_dm.num_entity_dofs(D+1) > 0 and V_dm.num_entity_dofs(0) == 0:
            bdofs = boundary_dofs[Vb_dm.tabulate_entity_dofs(D,0)]
            cdofs = cell_dofs[V_dm.tabulate_entity_dofs(D+1,0)]
            for bdof, cdof in zip(bdofs, cdofs):
                #if dolfin_version() in ["1.4.0+", "1.5.0"]:
                if LooseVersion(dolfin_version()) > LooseVersion("1.4.0"):
                    bdof = Vb_dm.local_to_global_index(bdof)
                    cdof = V_dm.local_to_global_index(cdof)

                dofmap_to_boundary[bdof] = cdof

    return dofmap_to_boundary

if __name__ == '__main__':
    from cbcpost.utils.utils import get_set_vector
    from dolfin import *
    import numpy as np
    #mesh = UnitCubeMesh(3,3,3)
    mesh = UnitSquareMesh(24,24)
    V = VectorFunctionSpace(mesh, "CG", 1)
    #V = FunctionSpace(mesh, "DG", 0)
    bmesh = BoundaryMesh(mesh, "exterior")
    Vb = VectorFunctionSpace(bmesh, "CG", 1)
    #Vb = FunctionSpace(bmesh, "DG", 0)


    dm = V.dofmap()
    dmb = Vb.dofmap()

    #mapping = mesh_to_boundarymesh_dofmap(bmesh, V, Vb)
    #keys = np.array(mapping.keys(), dtype=np.intc)
    #values = np.array(mapping.values(), dtype=np.intc)
    D = V
    Db = Vb
    local_dofmapping = mesh_to_boundarymesh_dofmap(bmesh, D, Db)
    #values, keys = zip(*local_dofmapping.iteritems())
    keys, values = zip(*local_dofmapping.iteritems())
    values = np.array(values, dtype=np.intc)
    keys = np.array(keys, dtype=np.intc)

    t = 3.0
    #expr = Expression("1+x[0]*x[1]*t", degree=1, t=t)
    expr = Expression(("1+x[0]*t", "3+x[1]*t"), degree=1, t=t)
    #expr = Expression(("1+x[0]*t", "3+x[1]*t", "10+x[2]*t"), degree=1, t=t)

    u = interpolate(expr, V)
    ub = interpolate(expr, Vb)

    #print MPI.sum(mpi_comm_world(), len(mapping.keys()))
    #print MPI.sum(mpi_comm_world(), len(mapping.values()))

    print(assemble(sqrt(inner(u,u))*ds))
    print(assemble(sqrt(inner(ub,ub))*dx))

    ub2 = Function(Vb)
    """
    for i in range(2):
        if i == MPI.rank(mpi_comm_world()):
            print "Process number: ", i
            print len(mapping.keys()), mapping.keys()
            print len(mapping.values()), mapping.values()
            print "Ownership range: ", dm.ownership_range()
            print "Bdry ownership range: ", dmb.ownership_range()
            keys = [k-dmb.ownership_range()[0] for k in mapping.keys()]
            values = [v-dm.ownership_range()[0] for v in mapping.values()]
            #print keys
            #print values
            #print ub2.vector()[keys]
            #print u.vector()[values]
        MPI.barrier(mpi_comm_world())
    exit()
    """
    #print dm.dofmap().ownership_range()
    #print ub.vector().get_local()
    #print len(ub.vector().get_local())
    #print MPI.sum(mpi_comm_world(), len(ub.vector().get_local()))
    #exit()
    #ub2.vector().__setitem__(keys, u.vector().__getitem__(values))
    #ub2.vector()[keys] = u.vector()[values]
    print("Keys: ", keys)
    print("Values: ", values)
    u2 = Function(V)
    #get_set_vector(ub2.vector(), keys, u.vector(), values)
    get_set_vector(u2.vector(), values, ub.vector(), keys)

    print(assemble(sqrt(inner(ub,ub))*dx))
    print(assemble(sqrt(inner(u2,u2))*ds))
    #plot(ub2)
    #plot(ub)
    #interactive()

    #print erroro
    File("ub.pvd") << ub
    File("ub2.pvd") << ub2


