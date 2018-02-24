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
"""Fields that require input parameters. This is typically fields that
can be used on different fields, for example time derivatives, averages,
parts of fields etc.
"""
from .Operators import Add, Divide, Subtract, Multiply

# Fields that need input
_meta_fields_constant = [
    # Inspecting parts of fields
    "SubFunction",
    "PointEval",
    "Boundary",
    "Restrict",

    # Spatial analysis of other fields
    "Norm",
    "ErrorNorm",
    "DomainAvg",
    "DomainSD",
    "Magnitude",
    "Minimum",
    "Maximum",
    "Threshold",
    "Dot",
]
_meta_fields_time = [
    # Transient analysis of other fields
    "TimeIntegral",
    "TimeAverage",
    "TimeDerivative",
    # Operators
    #"Add",
    #"Subtract",
    #"Multiply",
    #"Divide",
    
    # Timekeeping
    "Time",
    ]

meta_fields = _meta_fields_constant+_meta_fields_time

