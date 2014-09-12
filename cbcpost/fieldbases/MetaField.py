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

from cbcpost.fieldbases.Field import Field

class MetaField(Field):
    def __init__(self, value, params=None, name="default", label=None):
        Field.__init__(self, params, name, label)
        self.valuename = value.name if isinstance(value, Field) else value
    
    @property
    def name(self):
        if self._name == "default":
            n = self.__class__.__name__+"_"+self.valuename
            if self.label: n += "-"+self.label
        else:
            n = self._name
        return n
        
    
    def after_last_compute(self, get):
        u = get(self.valuename)
        if u != "N/A":
            return self.compute(get)

