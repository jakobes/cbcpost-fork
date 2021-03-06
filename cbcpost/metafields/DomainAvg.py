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
"""Functionality for calculating the average of a specified domain."""

from cbcpost.fieldbases.MetaField import MetaField
from cbcpost.utils.utils import cbc_warning
from cbcpost.utils.mpi_utils import gather, broadcast
from dolfin import (assemble, dx, Function, Constant, Measure, MeshFunctionSizet, MeshFunctionDouble,
                    MeshFunctionBool, MeshFunctionInt, MeshFunction, dolfin_version)
import numpy as np
from distutils.version import LooseVersion

def _init_measure(measure="default", cell_domains=None, facet_domains=None, indicator=None):
    assert cell_domains is None or facet_domains is None, "You can't specify both cell_domains or facet_domains"

    if cell_domains is not None:
        assert isinstance(cell_domains, (MeshFunctionSizet, MeshFunctionInt))

    if facet_domains is not None:
        assert isinstance(facet_domains, (MeshFunctionSizet, MeshFunctionInt))

    if (cell_domains and indicator is not None):
        if LooseVersion(dolfin_version()) > LooseVersion("1.6.0"):
            dI = Measure("cell")(subdomain_data=cell_domains)(indicator)
        else:
            dI = Measure("cell")[cell_domains](indicator)
    elif (facet_domains and indicator is not None):
        if LooseVersion(dolfin_version()) > LooseVersion("1.6.0"):
            dI = Measure("exterior_facet")(subdomain_data=facet_domains)(indicator)
        else:
            dI = Measure("exterior_facet")[facet_domains](indicator)
    elif measure == "default":
        if indicator is not None:
            cbc_warning("Indicator specified, but no domains. Will dompute average over entire domain.")
        dI = dx()
    elif isinstance(measure, Measure):
        dI = measure
    else:
        raise TypeError("Unable to create a domain measure from provided domains or measure.")

    return dI

def _init_label(measure):
    if measure.integral_type() == "cell" and measure.subdomain_id() == "everywhere":
        return None

    if measure.integral_type() == "cell":
        label = "dx"
    elif measure.integral_type() == "exterior_facet":
        label = "ds"
    elif measure.integral_type() == "interior_facet":
        label = "dS"
    else:
        label = measure.integral_type()

    if measure.subdomain_id() != "everywhere":
        label += str(measure.subdomain_id())

    return label

def duplicate_meshfunction(mf, new_mesh):
    "Duplicate meshfunction defined on a different mesh"
    # Rebuild meshfunction
    mesh = mf.mesh()
    dim = mf.dim()

    if isinstance(mf, MeshFunctionSizet):
        f = MeshFunction("size_t", new_mesh, dim)
        dtype = np.uintp
    elif isinstance(mf, MeshFunctionDouble):
        f = MeshFunction("double", new_mesh, dim)
        dtype = np.float_
    elif isinstance(mf, MeshFunctionBool):
        f = MeshFunction("bool", new_mesh, dim)
        dtype = np.bool
    elif isinstance(mf, MeshFunctionInt):
        f = MeshFunction("int", new_mesh, dim)
        dtype = np.int

    # Midpoint of old mesh
    connectivity = mesh.topology()(dim,0)().reshape(mesh.num_entities(dim), -1)
    midpoints = np.mean(mesh.coordinates()[connectivity], axis=1)
    indices = np.lexsort(midpoints.T[::-1])

    sorted_midpoints = midpoints[indices]
    values = mf.array()[indices]

    gdim = midpoints.shape[1]
    sorted_midpoints = sorted_midpoints.flatten()
    sorted_midpoints = gather(sorted_midpoints, 0)
    sorted_midpoints = np.hstack(sorted_midpoints).flatten()

    sorted_midpoints = broadcast(sorted_midpoints, 0)
    _sorted_midpoints = []

    i = 0
    for arr in sorted_midpoints:
        _sorted_midpoints.append(arr)
    sorted_midpoints = np.reshape(sorted_midpoints, newshape=(-1, gdim))
    values = gather(values, 0)
    values = np.hstack(values).flatten()

    values = broadcast(values, 0)

    # Workaround for bug in np.uint64
    if dtype==np.uintp:
        M = max(values)
        if M == dtype(-1):

            values[values == M] = -1

    values = values.astype(dtype)
    _values = []
    for arr in values:
        _values.append(arr)
    values = np.array(_values, dtype=dtype)

    indices = np.lexsort(sorted_midpoints.T[::-1])
    values = values[indices]
    sorted_midpoints = sorted_midpoints[indices]

    # Now has full list of midpoint+values on all processes

    # Sort midpoints on new mesh on current process
    connectivity = new_mesh.topology()(dim,0)().reshape(new_mesh.num_entities(dim), -1)
    midpoints = np.mean(new_mesh.coordinates()[connectivity], axis=1)
    indices = np.lexsort(midpoints.T[::-1])
    midpoints = midpoints[indices]

    idx = 0
    omp = sorted_midpoints[idx]
    tol = 1e-8

    for i, mp in enumerate(midpoints):
        for j in xrange(gdim):
            while omp[j] < mp[j]-tol:
                idx += 1
                omp[:] = sorted_midpoints[idx]
        f[int(indices[i])] = values[idx]

    return f


class DomainAvg(MetaField):
    """Compute the domain average for a specified domain. Default to computing
    the average over the entire domain.

    Parameters used to describe the domain are:

    :param measure: Measure describing the domain (default: dx())
    :param  cell_domains: A CellFunction describing the domains
    :param facet_domains: A FacetFunction describing the domains
    :param indicator: Domain id corresponding to cell_domains or facet_domains

    If cell_domains/facet_domains and indicator given, this overrides given measure.
    """
    def __init__(self, value, params=None, name="default", label=None, measure="default", cell_domains=None, facet_domains=None, indicator=None):
        self.dI = _init_measure(measure, cell_domains, facet_domains, indicator)
        if label is None:
            label = _init_label(self.dI)
        MetaField.__init__(self, value, params, name, label)

    def compute(self, get):
        u = get(self.valuename)

        if u is None:
            return

        if LooseVersion(dolfin_version()) > LooseVersion("1.6.0"):
            rank = len(u.ufl_shape)
            shape = u.ufl_shape
            mesh = u.ufl_domain()._ufl_cargo
            dIdomain = self.dI.ufl_domain()
        else:
            rank = u.rank()
            shape = u.shape()
            mesh = u.domain().data()
            dIdomain = self.dI.domain()

        mesh_id = mesh.id()

        # Find mesh/domain
        if isinstance(self.dI.subdomain_data(), MeshFunctionSizet):
            mf = self.dI.subdomain_data()
            mf_mesh_id = mf.mesh().id()

            if mf_mesh_id != mesh_id:
                mf = duplicate_meshfunction(mf, mesh)
            mesh = mf.mesh()

            self.dI = self.dI.reconstruct(domain=mesh, subdomain_data=mf)

        if dIdomain is None:
            self.dI = self.dI.reconstruct(domain=mesh)
            if LooseVersion(dolfin_version()) > LooseVersion("1.6.0"):
                dIdomain = self.dI.ufl_domain()
            else:
                dIdomain = self.dI.domain()

        assert dIdomain is not None

        # Calculate volume
        if not hasattr(self, "volume"):
            self.volume = assemble(Constant(1)*self.dI)
            assert self.volume > 0

        if rank == 0:
            value = assemble(u*self.dI)/self.volume
        elif rank == 1:
            value = [assemble(u[i]*self.dI)/self.volume for i in xrange(u.value_size())]
        elif rank == 2:
            value = []
            for i in xrange(shape[0]):
                for j in xrange(shape[1]):
                    value.append(assemble(u[i,j]*self.dI)/self.volume)

        return value
