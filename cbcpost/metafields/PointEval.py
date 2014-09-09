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
from cbcflow.post.fieldbases.MetaField import MetaField
import numpy as np
from dolfin import Point
from itertools import chain

def import_fenicstools():
    import fenicstools
    return fenicstools

def points_in_square(center, radius, resolution):
    points = []
    for i in range(resolution):
        for j in range(resolution):
            x = [center[0] + (i-(resolution-1.0)/2.0)*radius/(resolution-1.0),
                 center[1] + (j-(resolution-1.0)/2.0)*radius/(resolution-1.0)]
            points.append(tuple(x))
    return tuple(points)

def points_in_box(center, radius, resolution):
    points = []
    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                x = [center[0] + (i-(resolution-1.0)/2.0)*radius/(resolution-1.0),
                     center[1] + (j-(resolution-1.0)/2.0)*radius/(resolution-1.0),
                     center[2] + (k-(resolution-1.0)/2.0)*radius/(resolution-1.0)]
                points.append(tuple(x))
    return tuple(points)

def points_in_circle(center, radius, resolution):
    points = []
    for i in range(resolution):
        for j in range(resolution):
            x = [center[0] + (i-(resolution-1.0)/2.0)*radius/(resolution-1.0),
                 center[1] + (j-(resolution-1.0)/2.0)*radius/(resolution-1.0)]
            r2 = (x[0]-center[0])**2 + (x[1]-center[1])**2
            if r2 <= radius**2 + 1e-14:
                points.append(tuple(x))
    return tuple(points)

def points_in_ball(center, radius, resolution):
    points = []
    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                x = [center[0] + (i-(resolution-1.0)/2.0)*radius/(resolution-1.0),
                     center[1] + (j-(resolution-1.0)/2.0)*radius/(resolution-1.0),
                     center[2] + (k-(resolution-1.0)/2.0)*radius/(resolution-1.0)]
                r2 = (x[0]-center[0])**2 + (x[1]-center[1])**2 + (x[2]-center[2])**2
                if r2 <= radius**2 + 1e-14:
                    points.append(tuple(x))
    return tuple(points)

class PointEval(MetaField):
    def __init__(self, value, points, params=None, label=None):
        MetaField.__init__(self, value, params, label)
        self.points = points
        self._ft = import_fenicstools()

    @property
    def name(self):
        n = "%s_%s" % (self.__class__.__name__, self.valuename)
        if self.label: n += "_%s" %self.label
        return n

    def before_first_compute(self, get):
        u = get(self.valuename)

        # Convert 'Point' instances (not necessary if we
        # just assume tuples as input anyway)
        #dim = spaces.d
        if u == None:
            return
        dim = u.function_space().mesh().geometry().dim()
        self.coords = []
        for p in self.points:
            if isinstance(p, Point):
                pt = tuple((p.x(), p.y(), p.z())[:dim])
            else:
                pt = tuple(p[:dim])
            assert len(pt) == dim
            self.coords.append(pt)
        self.coords = tuple(self.coords)

        # Create Probes object (from fenicsutils)
        flattened_points = np.array(list(chain(*self.coords)), dtype=np.float)
        V = u.function_space()
        self.probes = self._ft.Probes(flattened_points, V)
        self._probetimestep = 0

        # This data is currently stored in the metadata file under 'init_data'
        return self.coords

    def compute(self, get):
        # Get field to probe
        u = get(self.valuename)

        # Evaluate in all points
        self.probes(u)

        # Fetch array with probe values at this timestep
        #results = self.probes.array(self._probetimestep)
        results = self.probes.array()
        self.probes.clear()
        #self._probetimestep += 1

        # Return as list to store without 'array(...)' text.
        # Probes give us no data if not on master node, so we just
        # return dummy list which is not used by postprocessor anyway.
        if results is None:
            return []
        else:
            if u.shape():
                return list(tuple(res) for res in results)
            else:
                return list(results)

    def after_last_compute(self, get):
        # This data is currently stored in the metadata file under 'finalize_data':
        return None
