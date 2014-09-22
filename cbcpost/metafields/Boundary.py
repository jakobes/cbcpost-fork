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
from cbcpost.fieldbases.MetaField import MetaField
from cbcpost.utils.mesh_to_boundarymesh_dofmap import mesh_to_boundarymesh_dofmap
from cbcpost import SpacePool

from dolfin import Function, ds, dx, assemble

class Boundary(MetaField):

    def before_first_compute(self, get):
        u = get(self.valuename)
        
        assert isinstance(u, Function), "Can only extract boundary values of Function-objects"
        
        FS = u.function_space()
        
        spaces = SpacePool(FS.mesh())
        element = FS.ufl_element()
        #FS_boundary = spaces.get_space(FS.ufl_element().degree(), FS.num_sub_spaces(), boundary=True)
        FS_boundary = spaces.get_custom_space(element.family(), element.degree(), element.value_shape(), boundary=True)
        
        local_dofmapping = mesh_to_boundarymesh_dofmap(spaces.BoundaryMesh, FS, FS_boundary)
        self._keys = local_dofmapping.keys()
        self._values = local_dofmapping.values()
        
        self.u_bdry = Function(FS_boundary)
    
    def compute(self, get):
        u = get(self.valuename)
        self.u_bdry.vector()[self._keys] = u.vector()[self._values]       

        return self.u_bdry
