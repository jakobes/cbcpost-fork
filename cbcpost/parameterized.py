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
Parameterized represents a suggested interface to create classes/objects with
associated parameters.
"""
from __future__ import division
from cbcpost import get_parse_command_line_arguments
import sys

#pylint: disable=R0921
class Parameterized(object):
    "Core functionality for parameterized subclassable components."
    def __init__(self, params):
        self.params = self.default_params()

        self.params.replace(params)
        if get_parse_command_line_arguments():
            args = sys.argv[1:]
            self.params.parse_args(args)

        # Assert for each subclass that we have all keys,
        # i.e. no default_params functions have been skipped
        # in the inheritance chain
        pkeys = set(self.params.keys())
        for cls in type(self).mro()[:-2]: # Skip object and Parameterized
            assert len(set(cls.default_params().keys()) - pkeys) == 0

    # --- Default parameter functions ---

    @classmethod
    def default_params(cls):
        "Merges base and user params into one ParamDict."
        raise NotImplementedError("Missing default_params implementation for \
                                  class %s" % (cls,))

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
        doc = cls.__doc__
        if doc is None:
            return "Missing description."
        else:
            return doc.split('\n')[0]

    def __str__(self):
        return "%s: %s" % (self.shortname(), self.description())
