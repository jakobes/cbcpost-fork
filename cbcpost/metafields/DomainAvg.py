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
from dolfin import assemble, dx, Function, Constant

class DomainAvg(MetaField):
    def compute(self, get):
        u = get(self.valuename)
        
        if u == None:
            return

        # Find mesh/domain
        if isinstance(u, Function):
            mesh = u.function_space().mesh()
        else:
            mesh = problem.mesh
        
        # Calculate volume
        if not hasattr(self, "volume"):
            self.volume = assemble(Constant(1)*dx(), mesh=mesh)
        
        if u.rank() == 0:
            value = assemble(u*dx(), mesh=mesh)/self.volume
        elif u.rank() == 1:
            value = [assemble(u[i]*dx(), mesh=mesh)/self.volume for i in xrange(u.value_size())]
        elif u.rank() == 2:
            value = []
            for i in xrange(u.shape()[0]):
                for j in xrange(u.shape()[1]):
                    value.append(assemble(u[i,j]*dx(), mesh=mesh)/self.volume)

        return value
