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
from dolfin import assemble, dx, Function, Constant, Measure

class DomainAvg(MetaField):
    def __init__(self, value, params=None, name="default", label=None, measure=dx(), cell_domains=None, facet_domains=None, indicator=None):
        assert cell_domains == None or facet_domains == None, "You can't specify both cell_domains or facet_domains"
        
        if (cell_domains and indicator != None):
            self.dI = Measure("cell")[cell_domains](indicator)
        elif (facet_domains and indicator != None):
            self.dI = Measure("exterior_facet")[facet_domains](indicator)
        else:
            if indicator != None:
                cbcwarning("Indicator specified, but no domains. Will dompute average over entire domain.")
            self.dI = measure
        
        if label == None and str(self.dI) != "dxeverywhere":
            label = str(self.dI)[:2]
            if self.dI.domain_id() != "everywhere":
                label += str(self.dI.domain_id())
        MetaField.__init__(self, value, params, name, label)
        
    
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
            self.volume = assemble(Constant(1)*self.dI, mesh=mesh)
            assert self.volume > 0
        
        if u.rank() == 0:
            value = assemble(u*self.dI, mesh=mesh)/self.volume
        elif u.rank() == 1:
            value = [assemble(u[i]*self.dI, mesh=mesh)/self.volume for i in xrange(u.value_size())]
        elif u.rank() == 2:
            value = []
            for i in xrange(u.shape()[0]):
                for j in xrange(u.shape()[1]):
                    value.append(assemble(u[i,j]*self.dI, mesh=mesh)/self.volume)

        return value
