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
"""
Compute the time spent from call to before_first_compute and after_last_compute.
"""
from cbcpost.fieldbases.Field import Field
from time import time

class Time(Field):
    """ Compute the time spent between before_first_compute and after_last_compute calls.

    Useful for crude time measuring.
    """
    @classmethod
    def default_params(cls):
        params = Field.default_params()
        params.update(
            finalize=True,
            )
        return params

    def before_first_compute(self, get):
        self.t1 = time()
        return None

    def compute(self, get):
        return None

    def after_last_compute(self, get):
        return time()-self.t1