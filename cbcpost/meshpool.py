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

from dolfin import Mesh, Expression, CellVolume, dx, assemble, Constant, dot
import gc, weakref
import numpy as np

class MeshPool(Mesh):
    """A mesh pool to reuse meshes across a program.
    FIXME: Mesh doesn't support weakref. id refers to the shared_ptr, not the actual object.
    """
    #_existing = weakref.WeakValueDictionary()
    _existing = dict()
    _X = [Expression("x[0]"),
          Expression(("x[0]", "x[1]")),
          Expression(("x[0]", "x[1]", "x[2]"))]

    #def __new__(cls, mesh, tolerance=1e-12):
    def __new__(cls, mesh, tolerance=1e-12):
        dim = mesh.geometry().dim()
        X = MeshPool._X[dim-1]

        test = assemble(dot(X,X)*CellVolume(mesh)**(0.25)*dx())*assemble(Constant(1)*dx(domain=mesh))
        assert test > 0.0
        assert test < np.inf

        # Do a garbage collect to collect any garbage references
        # Needed for full parallel compatibility
        gc.collect()
        keys = np.array(MeshPool._existing.keys())
        self = None
        if len(keys) > 0:
            diff = np.abs(keys-test)/np.abs(test)
            idx = np.argmin(np.abs(keys-test))

            if diff[idx] <= tolerance and isinstance(mesh, type(MeshPool._existing[keys[idx]])):
                self = MeshPool._existing[keys[idx]]

        if self == None:
            self = mesh
            MeshPool._existing[test] = self
        return self