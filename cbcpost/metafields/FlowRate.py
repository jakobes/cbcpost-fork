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

from cbcpost.fieldbases.Field import Field
from dolfin import assemble, dot

class FlowRate(Field):
    def __init__(self, boundary_id, params=None, label=None):
        Field.__init__(self, params, label)
        self.boundary_id = boundary_id

    @property
    def name(self):
        n = "%s_%s" % (self.__class__.__name__, self.boundary_id)
        if self.label: n += "_"+self.label
        return n

    def compute(self, get):
        u = get("Velocity")

        n = problem.mesh.ufl_cell().n
        dsi = problem.ds(self.boundary_id)

        M = dot(u, n)*dsi
        return assemble(M)
