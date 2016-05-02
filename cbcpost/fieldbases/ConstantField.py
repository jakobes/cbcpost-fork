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
"""Helper class for setting a constant field."""
from cbcpost.fieldbases.Field import Field

class ConstantField(Field):
    """Class for setting constant values. Helpful in dependency inspection."""
    def __init__(self, value, *params, **kwparams):
        Field.__init__(self, *params, **kwparams)
        self.value = value
        #if self.label is None:
        #    self.label = str(self.value)

    @property
    def name(self):
        """Return name of field. By default this is *classname_valuename-label*,
        but can be overloaded in subclass.
        """
        return str(self.value)

    def compute(self, get):
        return self.value