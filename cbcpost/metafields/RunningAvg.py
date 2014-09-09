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
from cbcflow.post.fieldbases.MetaField import MetaField
from dolfin import Function

class RunningAvg(MetaField):
    def before_first_compute(self, get):
        self._value = None
        self._count = 0

    def compute(self, get):
        u = get(self.valuename)

        self._count += 1

        if self._value is None:
            if isinstance(u, Function):
                self._sum = Function(u)
                self._value = Function(u)
            else:
                self._sum = u
                self._value = u
        else:
            if isinstance(u, Function):
                self._sum.vector().axpy(1.0, u.vector())
                self._value.vector().zero()
                self._value.vector().axpy(1.0/self._count, self._sum.vector())
            else:
                self._sum += u
                self._value = u / self._count

        return self._value

    def after_last_compute(self, get):
        return self._value
