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
"""Functionality for computing domain where value is smaller or greater than a treshold."""
from cbcpost import MetaField2
from dolfin import Function

class Threshold(MetaField2):
    """Compute a new Function based on input function and input threshold.
    Returned Function is 1 where above/below threshold, and 0 otherwise.
    """
    def __init__(self, *params, **kwparams):
        MetaField2.__init__(self, *params, **kwparams)
        assert self.params.threshold_by in ["below", "above"]

    @classmethod
    def default_params(cls):
        """
        Default parameters are:

        +----------------------+-----------------------+-------------------------------------------------------------------------------------------+
        |Key                   | Default value         |  Description                                                                              |
        +======================+=======================+===========================================================================================+
        | threshold_by         | "below"               | Set the function to threshold "above" or "below" threshold function                       |
        +----------------------+-----------------------+-------------------------------------------------------------------------------------------+
        """
        params = MetaField2.default_params()
        params.update(threshold_by="below")
        return params

    def compute(self, get):
        u = get(self.valuename1)
        threshold = get(self.valuename2)

        if u is None:
            return
        if isinstance(u, Function):
            if not hasattr(self, "u"):
                self.u = Function(u)
            if self.params.threshold_by == "below":
                self.u.vector()[:] = u.vector().array()<threshold
            elif self.params.threshold_by == "above":
                self.u.vector()[:] = u.vector().array()>threshold
            return self.u
        else:
            if self.params.threshold_by == "below":
                return u<threshold
            elif self.params.threshold_by == "above":
                return u>threshold