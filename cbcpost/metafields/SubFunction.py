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

from cbcpost.fieldbases.Field import Field
from dolfin import Function, VectorFunctionSpace, FunctionSpace, project, as_vector, MPI

def import_fenicstools():
    import fenicstools
    return fenicstools

class SubFunction(Field):
    "SubFunction is used to interpolate a Field on a non-matching mesh"
    def __init__(self, field, mesh, params=None, label=None):
        Field.__init__(self, params, label)
        
        import imp
        try:
            imp.find_module("mpi4py")
        except:
            raise ImportError("Can't find module mpi4py. This is required for SubFunction.")

        self._ft = import_fenicstools()

        self.mesh = mesh

        # Store only name, don't need the field
        if isinstance(field, Field):
            field = field.name
        self.valuename = field

    @property
    def name(self):
        n = "SubFunction_%s" % self.valuename
        if self.label: n += "_"+self.label
        return n

    def before_first_compute(self, get):
        u = get(self.valuename)
        
        V = u.function_space()
        element = V.ufl_element()        
        family = element.family()
        degree = element.degree()
        
        if u.rank() == 1:
            FS = VectorFunctionSpace(self.mesh, family, degree)
            FS_scalar = FS.sub(0).collapse()
            self.us = Function(FS_scalar)

        elif u.rank() == 0:
            FS = FunctionSpace(self.mesh, family, degree)
        else:
            raise Exception("Does not support TensorFunctionSpace yet")
        
        self.u = Function(FS, name=self.name)

    def compute(self, get):
        u = get(self.valuename)

        if u.rank() == 1:
            u = u.split()
            U = []
            for _u in u:
                U.append(self._ft.interpolate_nonmatching_mesh(_u, self.us.function_space()))

            self.u.assign(project(as_vector(U), self.u.function_space()))

        elif u.rank() == 0:
            U = self._ft.interpolate_nonmatching_mesh(u, self.u.function_space())
            self.u.assign(project(U, self.u.function_space()))
        return self.u
