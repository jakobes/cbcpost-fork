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
"""Functionality to implement operators on two fields"""
from cbcpost.fieldbases.MetaField2 import MetaField2
from cbcpost.fieldbases.ConstantField import ConstantField
from cbcpost.fieldbases.Field import Field

class OperatorField(MetaField2):
    "Base class for all operators on fields"
    def __init__(self, value1, value2, *params, **kwparams):
        # We need to wrap scalar or array values in a ConstantField
        # to avoid dependency errors
        self._add_fields = []
        if not isinstance(value1, (str, Field)):
            value1 = ConstantField(value1)
            self._add_fields.append(value1)
        if not isinstance(value2, (str, Field)):
            value2 = ConstantField(value2)
            self._add_fields.append(value2)

        MetaField2.__init__(self, value1, value2, *params, **kwparams)

    def add_fields(self):
        return self._add_fields

