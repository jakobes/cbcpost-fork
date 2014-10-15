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
"""Functionality for calculating the average of a specified domain."""

from cbcpost.fieldbases.MetaField import MetaField
from cbcpost.utils.utils import cbc_warning
from dolfin import assemble, dx, Function, Constant, Measure


class DomainAvg(MetaField):
    """Compute the domain average for a specified domain. Default to computing
    the average over the entire domain.

    Parameters used to describe the domain are:

    :param measure: Measure describing the domain (default: dx())
    :param  cell_domains: A CellFunction describing the domains
    :param facet_domains: A FacetFunction describing the domains
    :param indicator: Domain id corresponding to cell_domains or facet_domains

    If cell_domains/facet_domains and indicator given, this overrides given measure.
    """
    def __init__(self, value, params=None, name="default", label=None, measure=dx(), cell_domains=None, facet_domains=None, indicator=None):
        assert cell_domains == None or facet_domains == None, "You can't specify both cell_domains or facet_domains"

        if (cell_domains and indicator != None):
            self.dI = Measure("cell")[cell_domains](indicator)
        elif (facet_domains and indicator != None):
            self.dI = Measure("exterior_facet")[facet_domains](indicator)
        else:
            if indicator != None:
                cbc_warning("Indicator specified, but no domains. Will dompute average over entire domain.")
            self.dI = measure

        if label == None and not (self.dI.integral_type() == "cell" and self.dI.subdomain_id() == "everywhere"):

            if self.dI.integral_type() == "cell":
                label = "dx"
            elif self.dI.integral_type() == "exterior_facet":
                label = "ds"
            elif self.dI.integral_type() == "interior_facet":
                label = "dS"
            else:
                label = self.dI.integral_type()

            if self.dI.subdomain_id() != "everywhere":
                label += str(self.dI.subdomain_id())
        MetaField.__init__(self, value, params, name, label)


    def compute(self, get):
        u = get(self.valuename)

        if u == None:
            return

        # Find mesh/domain
        if isinstance(u, Function):
            mesh = u.function_space().mesh()

        if not self.dI.domain():
            self.dI = self.dI.reconstruct(domain=mesh)
        assert self.dI.domain()

        # Calculate volume
        if not hasattr(self, "volume"):
            self.volume = assemble(Constant(1)*self.dI)
            assert self.volume > 0

        if u.rank() == 0:
            value = assemble(u*self.dI)/self.volume
        elif u.rank() == 1:
            value = [assemble(u[i]*self.dI)/self.volume for i in xrange(u.value_size())]
        elif u.rank() == 2:
            value = []
            for i in xrange(u.shape()[0]):
                for j in xrange(u.shape()[1]):
                    value.append(assemble(u[i,j]*self.dI)/self.volume)

        return value
