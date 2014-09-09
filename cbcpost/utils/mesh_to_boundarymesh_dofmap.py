# Copyright (C) 2010-2014 Simula Research Laboratory
#
# This file is part of CBCFLOW.
#
# CBCFLOW is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CBCFLOW is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with CBCFLOW. If not, see <http://www.gnu.org/licenses/>.

from dolfin import Cell, Facet, MPI
from numpy import where

def mesh_to_boundarymesh_dofmap(boundary, V, Vb):
    "Find the mapping between dofs on boundary and boundary dofs of full mesh"    
    assert V.ufl_element().family() == Vb.ufl_element().family()
    assert V.ufl_element().degree() == Vb.ufl_element().degree()
    
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
                #if v_idx in boundary.topology().shared_entities(0):
                    #print boundary.topology().shared_entities(0)[v_idx]
                    #if boundary.topology().shared_entities(0)[v_idx] == MPI.process_number():
                    #if MPI.process_number() in boundary.topology().shared_entities(0)[v_idx]:
                        #print "hei"
                        #continue
                
                mesh_v_idx = vertex_map[int(v_idx)]

                mesh_list_idx = where(mesh_cell.entities(0) == mesh_v_idx)[0][0]
                boundary_list_idx = where(boundary_cell.entities(0) == v_idx)[0][0]

                bdofs = boundary_dofs[Vb_dm.tabulate_entity_dofs(0, boundary_list_idx)]
                cdofs = cell_dofs[V_dm.tabulate_entity_dofs(0, mesh_list_idx)]

                for bdof, cdof in zip(bdofs, cdofs):
                    if not (V_dm.ownership_range()[0] <= cdof < V_dm.ownership_range()[1]):
                        continue
                    dofmap_to_boundary[bdof] = cdof

        if V_dm.num_entity_dofs(3) > 0 and V_dm.num_entity_dofs(0) == 0:
            bdofs = boundary_dofs[Vb_dm.tabulate_entity_dofs(2,0)]
            cdofs = cell_dofs[V_dm.tabulate_entity_dofs(3,0)]
            for bdof, cdof in zip(bdofs, cdofs):
                dofmap_to_boundary[bdof] = cdof

    return dofmap_to_boundary

    
    
    