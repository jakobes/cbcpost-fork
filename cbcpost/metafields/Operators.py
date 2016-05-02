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
"""Different basic operators subclassing OperatorField"""
from __future__ import division
from cbcpost.fieldbases.OperatorField import OperatorField

class Add(OperatorField):
    """Add two fields"""
    def compute(self, get):
        a = get(self.valuename1)
        b = get(self.valuename2)
        if a is None or b is None:
            return None
        return a+b

class Multiply(OperatorField):
    """Multiply two fields"""
    def compute(self, get):
        a = get(self.valuename1)
        b = get(self.valuename2)
        if a is None or b is None:
            return None
        return a*b
            
class Subtract(OperatorField):
    """Subtract two fields"""
    def compute(self, get):
        a = get(self.valuename1)
        b = get(self.valuename2)
        if a is None or b is None:
            return None
        return a-b

class Divide(OperatorField):
    """Divide two fields"""
    def compute(self, get):
        a = get(self.valuename1)
        b = get(self.valuename2)
        if a is None or b is None:
            return None
        return a/b
