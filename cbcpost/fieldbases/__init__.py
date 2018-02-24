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
""" Base classes for all post fields. """

from .Field import Field
from .SolutionField import SolutionField
from .MetaField import MetaField
from .MetaField2 import MetaField2
from .ConstantField import ConstantField
from .OperatorField import OperatorField

__all__ = ["Field", "SolutionField", "MetaField", "MetaField2", "OperatorField"]
