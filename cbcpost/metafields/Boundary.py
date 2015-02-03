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
"""Functionality for evaluating a field at the mesh boundary """
from cbcpost.fieldbases.MetaField import MetaField
#from cbcpost.utils.mesh_to_boundarymesh_dofmap import mesh_to_boundarymesh_dofmap
from cbcpost.utils import get_set_vector, mesh_to_boundarymesh_dofmap
from cbcpost import SpacePool

from dolfin import Function
import numpy as np

class Boundary(MetaField):
    """Extracts the boundary values of a Function and returns a Function object
    living on the equivalent FunctionSpace on boundary mesh.

    .. warning::
        Only CG1 and DG0 spaces currently functioning.

    """
    def before_first_compute(self, get):
        u = get(self.valuename)

        assert isinstance(u, Function), "Can only extract boundary values of Function-objects"

        FS = u.function_space()

        spaces = SpacePool(FS.mesh())
        element = FS.ufl_element()
        #FS_boundary = spaces.get_space(FS.ufl_element().degree(), FS.num_sub_spaces(), boundary=True)
        FS_boundary = spaces.get_custom_space(element.family(), element.degree(), element.value_shape(), boundary=True)

        local_dofmapping = mesh_to_boundarymesh_dofmap(spaces.BoundaryMesh, FS, FS_boundary)
        #self._keys = local_dofmapping.keys()
        self._keys = np.array(local_dofmapping.keys(), dtype=np.intc)
        #self._values = local_dofmapping.values()
        self._values = np.array(local_dofmapping.values(), dtype=np.intc)
        self._temp_array = np.zeros(len(self._keys), dtype=np.float_)
        self.u_bdry = Function(FS_boundary)

    def compute(self, get):
        u = get(self.valuename)

        # The simple __getitem__, __setitem__ has been removed in dolfin 1.5.0.
        # The new cbcpost-method get_set_vector should be compatible with 1.4.0 and 1.5.0.
        #self.u_bdry.vector()[self._keys] = u.vector()[self._values]
        get_set_vector(self.u_bdry.vector(), self._keys, u.vector(), self._values, self._temp_array)

        return self.u_bdry
