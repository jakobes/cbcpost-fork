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
from cbcpost.utils.restriction_map import restriction_map
from cbcpost.utils import cbc_warning

from dolfin import Function, FunctionSpace, VectorFunctionSpace, TensorFunctionSpace
from numpy import array, uint

class Restrict(MetaField):
    "Restrict is used to restrict a Field to a submesh of the mesh associated with the Field."
    def __init__(self, field, submesh, params={}, label=None):
        MetaField.__init__(self, field, params, label)
        
        self.submesh = submesh
        
    @property
    def name(self):
        n = "Restrict_%s" % self.valuename
        if self.label: n += "_"+self.label
        return n
    
    def compute(self, get):
        u = get(self.valuename)
        
        if u == None:
            return None
        
        if not isinstance(u, Function):
            cbc_warning("Do not understand how to handle datatype %s" %str(type(u)))
            return None
        
        #if not hasattr(self, "restriction_map"):
        if not hasattr(self, "keys"):
            V = u.function_space()
            element = V.ufl_element()        
            family = element.family()
            degree = element.degree()
            
            if u.rank() == 0: FS = FunctionSpace(self.submesh, family, degree)
            elif u.rank() == 1: FS = VectorFunctionSpace(self.submesh, family, degree)
            elif u.rank() == 2: FS = TensorFunctionSpace(self.submesh, family, degree)
            
            self.u = Function(FS)
            
            
            #self.restriction_map = restriction_map(V, FS)
            rmap = restriction_map(V, FS)
            self.keys = array(rmap.keys(), dtype=uint)
            self.values = array(rmap.values(), dtype=uint)
            
            
        #self.u.vector()[self.restriction_map.keys()] = u.vector()[self.restriction_map.values()]
        self.u.vector()[self.keys] = u.vector()[self.values]
        return self.u
        