# Copyright (C) 2010-2014 Simula Research Laboratory
#
# This file is part of CBCFLOW.
#
# CBCFLOW is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CBCFLOW is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with CBCFLOW. If not, see <http://www.gnu.org/licenses/>.

# Helper functionality
from spacepool import SpacePool, get_grad_space
from paramdict import ParamDict
from parameterized import Parameterized

# Core functionality
from postprocessor import PostProcessor
from restart import Restart
from replay import Replay

from cbcpost.fieldbases import Field
from cbcpost.fieldbases import SolutionField
from cbcpost.fieldbases import MetaField
from cbcpost.fieldbases import MetaField2

from cbcpost.metafields import meta_fields

for f in meta_fields:
    exec("from cbcpost.metafields.%s import %s" % (f, f))