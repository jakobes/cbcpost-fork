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
"""Functionality to operate on two different Fields."""
from cbcpost.fieldbases.Field import Field

class MetaField2(Field):
    """ Base class for all Fields that operate on two different Fields.

    :param value1: First Field or fieldname to operate on
    :param value2: Second Field or fieldname to operate on

    """
    def __init__(self, value1, value2, params=None, name="default", label=None):
        Field.__init__(self, params, name, label)
        self.valuename1 = value1.name if isinstance(value1, Field) else value1
        self.valuename2 = value2.name if isinstance(value2, Field) else value2


    @property
    def name(self):
        """Return name of field. By default this is *classname_valuename1_valuename2-label*,
        but can be overloaded in subclass.
        """
        if self._name == "default":
            n = "%s_%s_%s" % (self.__class__.__name__, self.valuename1, self.valuename2)
            if self.label: n += "_"+self.label
        else:
            n = self._name

        return n

    @name.setter
    def name(self, value):
        """Set name property"""
        self._name = value

    def compute(self, get):
        get(self.valuename1)
        get(self.valuename2)
        Field.compute(get)
