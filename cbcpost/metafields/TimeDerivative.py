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
"""Functionality for computing time derivatives."""
from cbcpost.fieldbases.MetaField import MetaField
from dolfin import Function

class TimeDerivative(MetaField):
    r"""Compute the time derivative of a Field :math:`F` through an explicit difference formula:

    .. math ::

        F'(t_n) \approx \frac{F(t_n)-F(t_{n-1})}{t_n-t_{n-1}}

    """
    def compute(self, get):
        u1 = get(self.valuename)
        u0 = get(self.valuename, -1)

        t1 = get("t")
        t0 = get("t", -1)
        dt = t1 - t0
        if dt == 0:
            dt = 1e-14 # Avoid zero-division

        if isinstance(u0, Function):
            # Create function to hold result first time,
            # assuming u1 and u0 are Functions in same space
            if not hasattr(self, "_du"):
                self._du = Function(u0.function_space())

            # Compute finite difference derivative # FIXME: Validate this, not tested!
            self._du.vector().zero()
            self._du.vector().axpy(+1.0/dt, u1.vector())
            self._du.vector().axpy(-1.0/dt, u0.vector())
            return self._du
        elif hasattr(u0, "__len__"):
            du = [(x1-x0) / dt for x0,x1 in zip(u0,u1)]
            du = type(u0)(du)
            return du
        else:
            # Assuming scalar value
            du = (u1 - u0) / dt
            return du
