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
"""Functionality for calculating the standard deviation of a specified domain."""
from cbcpost.fieldbases.MetaField import MetaField
from cbcpost.utils.utils import cbc_warning
from dolfin import assemble, Function, Constant, MeshFunction, project
from numpy import sqrt

from cbcpost.metafields.DomainAvg import DomainAvg, _init_label, _init_measure

class DomainSD(DomainAvg):
    """Compute the domain standard deviation for a specified domain. Default to computing
    the standard devitation over the entire domain.

    Parameters used to describe the domain are:

    :param measure: Measure describing the domain (default: dx())
    :param  cell_domains: A CellFunction describing the domains
    :param facet_domains: A FacetFunction describing the domains
    :param indicator: Domain id corresponding to cell_domains or facet_domains

    If cell_domains/facet_domains and indicator given, this overrides given measure.
    """
    def __init__(self, value, params=None, name="default", label=None, measure="default", cell_domains=None, facet_domains=None, indicator=None):
        DomainAvg.__init__(self, value, params, name, label, measure, cell_domains, facet_domains, indicator)

    def compute(self, get):
        u = get(self.valuename)
        
        if u == None:
            return
        
        assert isinstance(u, Function), "Unable to compute stdev of object of type %s" %type(u)

        # Compute the domain average using a dummy get-function passed to the DomainAvg compute
        ubar = DomainAvg.compute(self, lambda x,y=0: u)

        if isinstance(ubar, list):
            var = []
            for i in range(len(u)):
                vari = DomainAvg.compute(self, lambda x,y=0: (u[i]-Constant(ubar[i]))**2)
                var.append(vari)
        else:
            var = DomainAvg.compute(self, lambda x,y=0: (u-Constant(ubar))**2)

        if isinstance(var, (float, int, long)):
            stdev = sqrt(var)
        else:
            stdev = [sqrt(v) for v in var]
        
        return stdev
