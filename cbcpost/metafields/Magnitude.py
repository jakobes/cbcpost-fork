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
Compute the (piecewise) magnitude of a Field.
"""
from cbcpost.fieldbases.MetaField import MetaField
from dolfin import project, sqrt, Function, inner
from cbcpost.utils import cbc_warning

class Magnitude(MetaField):
    """ Compute the magnitude of a Function-evaluated Field.
    
    Supports function spaces where all subspaces are equal.
    """
    def before_first_compute(self, get):
        u = get(self.valuename)
        if isinstance(u, Function):
            if u.rank() == 0:
                self.f = Function(u.function_space())
            elif u.rank() >= 1:
                # Assume all subpaces are equal
                V = u.function_space().extract_sub_space([0]).collapse()
                self.f = Function(V)
        else:
            # Don't know how to handle object
            cbc_warning("Don't know how to calculate magnitude of object of type %s." %type(u))

    
    def compute(self, get):
        u = get(self.valuename)
        
        if isinstance(u, Function):
            if u.rank() == 0:
                self.f.vector().zero()
                self.f.vector().axpy(1.0, u.vector())
                self.f.vector().abs()
                return self.f
            elif u.rank() >= 1:
                self.f.assign(project(sqrt(inner(u,u)), self.f.function_space()))
                return self.f
        else:
            # Don't know how to handle object
            cbc_warning("Don't know how to calculate magnitude of object of type %s. Returning object." %type(u))
            return u
                        
    