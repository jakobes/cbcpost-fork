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
from dolfin import Function
import numpy

class RunningMax(MetaField):
    def before_first_compute(self, get):
        self._value = None

    def compute(self, get):
        u = get(self.valuename)

        if self._value is None:
            if isinstance(u, Function):
                self._value = u.copy(deepcopy=True)
            else:
                self._value = u
        else:
            if isinstance(u, Function):
                # TODO: Test! This might work at least in serial, what about paralell?
                self._value.vector()[:] = numpy.max(self._value.vector()[:], u.vector()[:])
            else:
                self._value = max(self._value, u)

        return self._value

    def after_last_compute(self, get):
        return self._value
