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
r'''
Computes the norm of two Fields. If the Fields Function-objects, the computation is forwarded to
the dolfin function *errornorm*. Otherwise two float list-type object is expected, and the :math:'l^p'-norm is computed as

.. math:: ||\mathbf{x}||_p := \left( \sum_i=1^n |x_i|^p \right)^{1/p}.

The :math:'\infty'-norm is computed as

.. math:: ||\mathbf{x}||_\infty := max(|x_1|, |x_2|, ..., |x_n|)

'''
from cbcpost.fieldbases.MetaField import MetaField
from dolfin import Function, Vector, norm

class ErrorNorm(MetaField):
    @classmethod
    def default_params(cls):
        params = MetaField.default_params()
        params.update(
            norm_type='default',
            degree_rise='3',
            )
        return params
    
    @property
    def name(self):
        n = "%s" % (self.__class__.__name__)
        if self.params.norm_type != "default": n += "_"+self.params.norm_type
        n += "_"+self.valuename1+"_"+self.valuename2
        if self.label: n += "_"+self.label
        return n
    
    def compute(self, get):
        u = get(self.valuename1)
        uh = get(self.valuename2)
        
        if u == None:
            return None
        
        if isinstance(uh, Function):
            norm_type = self.params.norm_type if self.params.norm_type != "default" else "L2"
            return errornorm(u, uh, norm_type=norm_type, degree_rise=self.params.degree_rise)
        else:
            if isinstance(u, (int, long, float)):
                u = [u]
            if isinstance(uh, (int, long, float)):
                uh = [uh]
            
            assert hasattr(u, "__len__")
            assert hasattr(uh, "__len__")
            assert len(u) == len(uh)
            
            if self.params.norm_type == 'default':
                norm_type = 'l2'
            
            if self.params.norm_type == 'linf':
                return max([abs(_u-_uh) for _u,_uh in zip(u,uh)])
                
            else:
                # Extract norm type
                p = int(self.params.norm_type[1:])
                return sum(abs(_u-_uh)**p for _u, _uh in zip(u,uh))**(1./p)
