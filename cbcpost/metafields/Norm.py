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
r'''
Computes the norm of a Field. If the Field returns a Vector or Function, the computation is forwarded to
the dolfin function *norm*. Otherwise a float list-type object is expected, and the :math:'l^p'-norm is computed as

.. math:: ||\mathbf{x}||_p := \left( \sum_i=1^n |x_i|^p \right)^{1/p}.

The :math:'\infty'-norm is computed as

.. math:: ||\mathbf{x}||_\infty := max(|x_1|, |x_2|, ..., |x_n|)

'''
from cbcflow.post.fieldbases.MetaField import MetaField
from dolfin import Function, Vector, norm

class Norm(MetaField):
    @classmethod
    def default_params(cls):
        params = MetaField.default_params()
        params.update(
            norm_type='default',
            )
        return params
    
    @property
    def name(self):
        n = "%s" % (self.__class__.__name__)
        if self.params.norm_type != "default": n += "_"+self.params.norm_type
        n += "_"+self.valuename
        if self.label: n += "_"+self.label
        return n
    
    def compute(self, get):
        u = get(self.valuename)
        
        if u == None:
            return None
        
        norm_type = self.params.norm_type
        
        if isinstance(u, Function):
            norm_type = norm_type if norm_type != "default" else "L2"
            return norm(u, norm_type)
        elif isinstance(u, Vector):
            norm_type = norm_type if norm_type != "default" else "l2"
            return norm(u, norm_type)
        else:
            if isinstance(u, (int, long, float)):
                u = [u]
            
            assert hasattr(u, "__len__")
            
            if self.params.norm_type == 'default':
                norm_type = 'l2'
            
            if self.params.norm_type == 'linf':
                return max([abs(_u) for _u in u])
                
            else:
                # Extract norm type
                p = int(norm_type[1:])
                return sum(abs(_u)**p for _u in u)**(1./p)
