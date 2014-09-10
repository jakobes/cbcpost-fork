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
from __future__ import division

class Parameterized(object):
    "Core functionality for parameterized subclassable components."
    def __init__(self, params):
        self.params = self.default_params()
        self.params.replace(params)

        # Assert for each subclass that we have all keys,
        # i.e. no default_params functions have been skipped
        # in the inheritance chain
        ps = set(self.params.keys())
        for cl in type(self).mro()[:-2]: # Skip object and Parameterized
            assert len(set(cl.default_params().keys()) - ps) == 0

    # --- Default parameter functions ---

    @classmethod
    def default_params(cls):
        "Merges base and user params into one ParamDict."
        raise NotImplementedError("Missing default_params implementation for class %s" % (cls,))

    # --- Name functions ---

    @classmethod
    def shortname(cls):
        """Get a one-word description of what the class represents.

        By default uses class name."""
        return cls.__name__

    @classmethod
    def description(cls):
        """Get a one-sentence description of what the class represents.

        By default uses first line of class docstring."""
        d = cls.__doc__
        if d is None:
            return "Missing description."
        else:
            return d.split('\n')[0]

    def __str__(self):
        return "%s: %s" % (self.shortname(), self.description())
