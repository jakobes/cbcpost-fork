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
from cbcpost.fieldbases.MetaField import MetaField
from dolfin import project, sqrt, Function, inner
from cbcflow.utils.common import cbcflow_warning

class Magnitude(MetaField):
    def compute(self, get):
        u = get(self.valuename)
        
        if isinstance(u, Function):
            if u.rank() == 0:
                return u
            elif u.rank() >= 1:
                # Assume all subpaces are equal
                V = u.function_space().extract_sub_space([0]).collapse()
                mag = project(sqrt(inner(u,u)), V)
                return mag
        else:
            # Don't know how to handle object
            cbcflow_warning("Don't know how to calculate magnitude of object of type %s. Returning object." %type(u))
            return u
                        
    