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
Pooling of meshes helps replay when the same can be created from different files.
It also helps reduce memory consumption.
"""

from dolfin import Mesh, SpatialCoordinate, CellVolume, dx, assemble, Constant, dot
import gc, weakref
import numpy as np

class MeshPool(Mesh):
    "A mesh pool to reuse meshes across a program."
    _existing = weakref.WeakValueDictionary()

    def __new__(cls, mesh, tolerance=1e-8):
        X = SpatialCoordinate(mesh)
        test = assemble(dot(X,X)*CellVolume(mesh)*dx())*assemble(Constant(1)*dx(domain=mesh))
        
        # Do a garbage collect to collect any garbage references
        # Needed for full parallel compatibility
        gc.collect()

        keys = np.array(MeshPool._existing.keys())
        self = None
        if len(keys) > 0:
            diff = np.abs(keys-test)
            idx = np.argmin(np.abs(keys-test))
            if diff[idx] < tolerance:
                self = MeshPool._existing[keys[idx]]

        if self == None:
            self = mesh
            MeshPool._existing[test] = self

        return self