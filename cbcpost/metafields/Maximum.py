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
"""Calculates the maximum of a Field."""
from cbcpost.fieldbases.MetaField import MetaField
from dolfin import Function, MPI, mpi_comm_world
import numpy

class Maximum(MetaField):
    """Computes the maximum of a Field."""
    def compute(self, get):
        u = get(self.valuename)
        
        if u == None:
            return None
        
        if isinstance(u, Function):
            return MPI.max(mpi_comm_world(), numpy.max(u.vector().array()))
        elif hasattr(u, "__len__"):
            return MPI.max(mpi_comm_world(), max(u))
        elif isinstance(u, (float,int,long)):
            return MPI.max(mpi_comm_world(), u)
        else:
            raise Exception("Unable to take max of %s" %str(u))
        