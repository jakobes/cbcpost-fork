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
Pooling of function spaces to reduce computational cost and memory consumption for
function spaces used several places in a program.
"""

from dolfin import (FunctionSpace, VectorFunctionSpace, TensorFunctionSpace, BoundaryMesh,
                    grad, Coefficient, dolfin_version)
from distutils.version import LooseVersion
import weakref, gc

def galerkin_family(degree):
    "Returns CG if degree>0, else DG."
    return "CG" if degree > 0 else "DG"

def decide_family(family, degree):
    "Helper function to decide family."
    return galerkin_family(degree) if family == "auto" else family

def get_grad_space(u, family="auto", degree="auto", shape="auto"):
    """Get gradient space of Function.

        .. warning::
            This is experimental and currently only designed to work with CG-spaces.

        """
    V = u.function_space()
    spaces = SpacePool(V.mesh())
    return spaces.get_grad_space(V, family, degree, shape)


class SpacePool(object):
    "A function space pool to reuse spaces across a program."
    _existing = weakref.WeakValueDictionary()

    def __new__(cls, mesh):
        key = mesh.id()

        # Do a garbage collect to collect any garbage references
        # Needed for full parallel compatibility
        gc.collect()

        # Check if spacepool exists, or create
        self = SpacePool._existing.get(key)
        if self is None:
            self = object.__new__(cls)
            self._init(mesh)
            SpacePool._existing[key] = self

        return self

    def __init__(self, mesh):
        pass

    def _init(self, mesh):
        # Store mesh reference to create future spaces
        self.mesh = mesh

        # Get dimensions for convenience
        cell = mesh.ufl_cell()
        self.gdim = cell.geometric_dimension()
        self.tdim = cell.topological_dimension()
        self.gdims = range(self.gdim)
        self.tdims = range(self.tdim)

        # For compatibility, remove when code has been converted
        self.d = self.gdim
        self.dims = self.gdims

        # Start with empty cache
        self._spaces = {}

        self._boundary = None

    def get_custom_space(self, family, degree, shape, boundary=False):
        """Get a custom function space. """

        if boundary:
            mesh = self.BoundaryMesh
            key = (family, degree, shape, boundary)
        else:
            mesh = self.mesh
            key = (family, degree, shape)
        space = self._spaces.get(key)

        if space is None:
            rank = len(shape)
            if rank == 0:
                space = FunctionSpace(mesh, family, degree)
            elif rank == 1:
                space = VectorFunctionSpace(mesh, family, degree, shape[0])
            else:
                space = TensorFunctionSpace(mesh, family, degree, shape, symmetry={})
            self._spaces[key] = space

        return space

    def get_space(self, degree, rank, family="auto", boundary=False):
        """ Get function space. If family == "auto", Lagrange is chosen
        (DG if degree == 0).
        """
        family = decide_family(family, degree)
        shape = (self.gdim,)*rank
        return self.get_custom_space(family, degree, shape, boundary)

    def get_grad_space(self, V, family="auto", degree="auto", shape="auto"):
        """Get gradient space of FunctionSpace V.

        .. warning::
            This is experimental and currently only designed to work with CG-spaces.

        """
        element = V.ufl_element()

        if degree == "auto":
            degree = element.degree() - 1

        if family == "auto":
            family = "DG"

        if family in ("CG", "Lagrange") and degree == 0:
            family = "DG"

        if shape == "auto":
            if LooseVersion(dolfin_version()) > LooseVersion("1.6.0"):
                shape = grad(Coefficient(element)).ufl_shape
            else:
                shape = grad(Coefficient(element)).shape()

        DV = self.get_custom_space(family, degree, shape)
        return DV

    @property
    def BoundaryMesh(self):
        if self._boundary == None:
            self._boundary = BoundaryMesh(self.mesh, "exterior")
        return self._boundary
