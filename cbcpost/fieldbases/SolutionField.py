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
""" Field representing a Solution. """
from cbcpost.fieldbases.Field import Field

class SolutionField(Field):
    """Helper class to specify solution variables to the postprocessor.
    
    :param name: Name of the solution field
    
    This field can be added to the postprocessor, although it does not implement
    a *compute*-method. A solution with the same name is expected to be passed
    to the *PostProcessor.update_all*-method.
    """
    def __init__(self, name, params=None, label=None):
        Field.__init__(self, params, name, label)
        