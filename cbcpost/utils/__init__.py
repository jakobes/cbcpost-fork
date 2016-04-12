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

"A collection of utilities used across cbcpost. "

from utils import (import_fenicstools, on_master_process, in_serial, strip_code, hdf5_link,
                   safe_mkdir, Loadable, create_function_from_metadata, cbc_warning, cbc_log,
                   cbc_print, get_memory_usage, time_to_string, Timer, timeit, loadable_formats,
                   get_set_vector)
from restriction_map import restriction_map
from submesh import create_submesh
from mesh_to_boundarymesh_dofmap import mesh_to_boundarymesh_dofmap
from Slice import Slice
from connectivity import compute_connectivity


from inspect import getmodule as _getmodule
__all__ = [k for k,v in locals().items() if "cbcpost.utils" in getattr(_getmodule(v), "__name__", "")]
