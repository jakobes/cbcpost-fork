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
"Functionality for computing dot product"
from dolfin import GenericFunction, Function, dot, project, Constant, assemble, dx
from numpy import dot as npdot

class Dot(MetaField2):
    "Compute the dot product between two fields"
    def compute(self, get):
        u1 = get(self.valuename1)
        u2 = get(self.valuename2)

        if u1 == None or u2 == None:
            return

        if not (isinstance(u1, GenericFunction) or isinstance(u2, GenericFunction)):
            return npdot(u1,u2)

        if not isinstance(u1, GenericFunction):
            u1 = Constant(u1)
            u1,u2 = u2,u1
        if not isinstance(u2, GenericFunction):
            u2 = Constant(u2)

        if isinstance(u2, Function):
            u1,u2 = u2,u1

        assert isinstance(u1, Function)
        assert isinstance(u2, GenericFunction)

        if u1.value_rank() == u2.value_rank():
            if u1.value_rank() == 0:
                V = u1.function_space()
            else:
                V = u1.function_space().sub(0).collapse()
        elif u1.value_rank() > u2.value_rank():
            assert u2.value_rank() == 0
            V = u1.function_space()
            u1,u2 = u2,u1
        else:
            assert isinstance(u2, Function)
            assert u1.value_rank() == 0
            V = u2.function_space()

        N = max([u1.value_rank(), u2.value_rank()])

        if not hasattr(self, "u"):
            self.u = Function(V)

        if isinstance(u2, Function) and u1.function_space().dim() == u2.function_space().dim():
            self.u.vector()[:] = u1.vector().array()*u2.vector().array()
        elif u1.value_rank() == u2.value_rank():
            project(dot(u1,u2), function=self.u)
        else:
            assert u1.value_rank() == 0
            if isinstance(u1, Constant):
                self.u.vector()[:] = float(u1)*u2.vector().array()
            else:
                project(u1*u2, function=self.u)

        return self.u