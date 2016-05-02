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
""" Functionality for computing time integrals. """

from cbcpost.fieldbases.MetaField import MetaField
from dolfin import Function
EPS = 1e-10

class TimeIntegral(MetaField):
    r"""
    Compute a time integral of a field :math:`F` by the backward trapezoidal method:

    .. math::

        \int_{T0}^{T1} F dt \approx \sum_ {n=1}^{n=N} \frac{F(t_{n-1})+F(t_n)}{2} (t_{n-1}-t_n)

    where :math:`t_0 = T0` and :math:`t_N = T1`.
    """
    @classmethod
    def default_params(cls):
        params = MetaField.default_params()
        params.update(
            finalize=True,
            )
        return params

    def compute(self, get):
        t1 = get("t")
        if hasattr(self, "tprev"):
            t0 = self.tprev
        else:
            t0 = get("t", -1)
        #self.tprev = t1
        #t0 = get("t", -1)

        assert t0 <= self.params.end_time+EPS and t1 >= self.params.start_time-EPS, "Trying to compute integral outside the integration limits!"

        # Get integrand
        u1 = get(self.valuename)
        if hasattr(self, "uprev"):
            u0 = self.uprev
        else:
            u0 = get(self.valuename, -1)
            #u0 = u1
            #self.uprev = u0
        #u0 = get(self.valuename, -1)

        assert u0 != "N/A" and u1 != "N/A", "u0=%s, u1=%s" %(str(u0), str(u1))

        # Interpolate to integration limits, if t1 or t0 is outside
        if t0 < self.params.start_time-1e-14:
            if isinstance(u0, Function): start = u0.vector() + (u1.vector()-u0.vector())/(t1-t0)*(self.params.start_time-t0)
            elif hasattr(u0, "__len__"): start = [u0[i]+(u1[i]-u0[i])/(t1-t0)*(self.params.start_time-t0) for i in range(len(u0))]
            else: start = u0 + (u1-u0)/(t1-t0)*(self.params.start_time-t0)
            t0 = self.params.start_time
        else:
            if isinstance(u0, Function): start = u0.vector()
            else: start = u0

        if t1 > self.params.end_time:
            if isinstance(u0, Function): end = u0.vector() + (u1.vector()-u0.vector())/(t1-t0)*(self.params.end_time-t0)
            elif hasattr(u0, "__len__"): end = [u0[i]+(u1[i]-u0[i])/(t1-t0)*(self.params.end_time-t0) for i in range(len(u0))]
            else: end = u0 + (u1-u0)/(t1-t0)*(self.params.end_time-t0)
            t1 = self.params.end_time
        else:
            if isinstance(u1, Function): end = u1.vector()
            else: end = u1

        dt = t1 - t0
        if dt == 0: dt = 1e-14 # Avoid zero-division

        # Add to sum
        if isinstance(u0, Function):
            # Create placeholder for sum the first time
            if not hasattr(self, "_sum"):
                self._sum = Function(u0.function_space())
            # Accumulate using trapezoidal integration
            self._sum.vector().axpy(dt/2.0, start) # FIXME: Validate this, not tested!
            self._sum.vector().axpy(dt/2.0, end)
        elif hasattr(u0, "__len__"):
            # Create placeholder for sum the first time
            if not hasattr(self, "_sum"):
                self._sum = [0.0]*len(u0)

            # Accumulate using trapezoidal integration
            for i in range(len(u0)):
                self._sum[i] += dt/2.0*start[i]
                self._sum[i] += dt/2.0*end[i]
        else:
            # Create placeholder for sum the first time
            if not hasattr(self, "_sum"):
                self._sum = 0.0
            # Accumulate using trapezoidal integration
            self._sum += dt/2.0*start
            self._sum += dt/2.0*end

        #print "Integrating %s from %f to %f (start=%s, end=%s). sum=%s" %(self.valuename, t0, t1, start, end, str(self._sum))

        # Store bounds for sanity check
        if not hasattr(self, "T0"):
            self.T0 = t0
        self.T1 = t1

        self.tprev = t1
        self.uprev = u1
        #print "Integrated %s from %f to %f" %(self.valuename, self.T0, self.T1)

        if not self.params.finalize:
            return self._sum
        else:
            return None

    def after_last_compute(self, get):
        if not hasattr(self, "_sum"):
            return None

        # Integrate last timestep if integration not completed
        if self.T1 <= self.params.end_time - EPS:
            self.compute(get)

        #print "Integrated %s from %f (start_time=%f) to %f (end_time=%f) (result=%s)" %(self.valuename, self.T0, self.params.start_time, self.T1, self.params.end_time,  str(self._sum))

        return self._sum
