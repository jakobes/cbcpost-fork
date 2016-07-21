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
cbcpost is a postprocessing framework designed for time-dependent problems solved using the
FEniCS project.
"""

# Check that a "good" dbm-module exists
from anydbm import _defaultmod
assert _defaultmod.__name__ in ["dbm", "gdbm", "dbhash"], "Unable to find a required DBM-implementation"
del _defaultmod

_use_cmdline_args = False
def set_parse_command_line_arguments(option):
    "Switch on/off command line argument parsing"
    assert option in [0,1,True,False]
    global _use_cmdline_args
    _use_cmdline_args = option

def get_parse_command_line_arguments():
    "Return whether to parse command line arguments"
    return _use_cmdline_args


# Helper functionality
from cbcpost.spacepool import SpacePool, get_grad_space
from cbcpost.meshpool import MeshPool
from cbcpost.paramdict import ParamDict
from cbcpost.parameterized import Parameterized

# Core functionality
from cbcpost.postprocessor import PostProcessor
from cbcpost.saver import Saver
from cbcpost.planner import Planner
from cbcpost.plotter import Plotter

from cbcpost.restart import Restart
from cbcpost.replay import Replay

from cbcpost.fieldbases import Field
from cbcpost.fieldbases import SolutionField
from cbcpost.fieldbases import MetaField
from cbcpost.fieldbases import MetaField2
from cbcpost.fieldbases import OperatorField
from cbcpost.fieldbases import ConstantField

from cbcpost.metafields import meta_fields
from cbcpost.metafields.Operators import Add, Subtract, Multiply, Divide

for f in meta_fields:
    exec("from cbcpost.metafields.%s import %s" % (f, f))
del f
