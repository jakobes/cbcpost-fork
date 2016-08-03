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
"""Functionality to (spatially) restrict a Field."""
from cbcpost.fieldbases.MetaField import MetaField
from cbcpost.utils.restriction_map import restriction_map
from cbcpost.utils import cbc_warning, get_set_vector

from dolfin import (Function, FunctionSpace, VectorFunctionSpace, TensorFunctionSpace,
                    dolfin_version)
from distutils.version import LooseVersion
#from numpy import array, uint, intc
import numpy as np

class Restrict(MetaField):
    """Restrict is used to restrict a Field to a submesh of the
    mesh associated with the Field.

    .. warning::

        This has only been tested for CG spaces and DG spaces of degree 0.

    """
    def __init__(self, field, submesh, params={}, name="default", label=None):
        MetaField.__init__(self, field, params, name, label)
        self.submesh = submesh

    def compute(self, get):
        u = get(self.valuename)

        if u is None:
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

            if LooseVersion(dolfin_version()) > LooseVersion("1.6.0"):
                rank = len(u.ufl_shape)
            else:
                rank = u.rank()

            if rank == 0: FS = FunctionSpace(self.submesh, family, degree)
            elif rank == 1: FS = VectorFunctionSpace(self.submesh, family, degree)
            elif rank == 2: FS = TensorFunctionSpace(self.submesh, family, degree, symmetry={})

            self.u = Function(FS)


            #self.restriction_map = restriction_map(V, FS)
            rmap = restriction_map(V, FS)
            self.keys = np.array(rmap.keys(), dtype=np.intc)
            self.values = np.array(rmap.values(), dtype=np.intc)
            self.temp_array = np.zeros(len(self.keys), dtype=np.float_)

        # The simple __getitem__, __setitem__ has been removed in dolfin 1.5.0.
        # The new cbcpost-method get_set_vector should be compatible with 1.4.0 and 1.5.0.
        #self.u.vector()[self.keys] = u.vector()[self.values]

        get_set_vector(self.u.vector(), self.keys, u.vector(), self.values, self.temp_array)
        return self.u
