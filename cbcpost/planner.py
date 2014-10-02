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
Module to handle the planning of Field computations of cbcpost.

Plans for computations to be executed at future timesteps by inspecting and
trigger computation of dependent Fields when required. This is required for
time-dependent Fields, such as TimeDerivative or TimeIntegral.

This code is intended for internal usage, and is called from a PostProcessor
instance.
"""

from collections import defaultdict

class Planner():
    "Planner class to plan for all computations."
    def __init__(self, timer, initial_dt):
        self._timer = timer
        self._plan = defaultdict(lambda: defaultdict(int))
        
        self._t_prev = None
        self._initial_dt = initial_dt
        
        # Keep track of last (time, timestep) computation of each field was triggered directly
        self._last_trigger_time = defaultdict(lambda: (-1e16, -1e16))
    
    def _should_compute_at_this_time(self, field, t, timestep):
        "Check if field is to be computed at current time"
        # If we later wish to move configuration of field compute frequencies to NSPostProcessor,
        # it's easy to swap here with e.g. fp = self._field_params[field.name]
        fp = field.params

        # Limit by timestep interval
        s = fp.start_timestep
        e = fp.end_timestep
        if not (s <= timestep <= e):
            return False

        # Limit by time interval
        s = fp.start_time
        e = fp.end_time
        eps = 1e-10
        if not (s-eps <= t <= e+eps):
            return False

        # Limit by frequency (accept if no previous data)
        pct, pcts = self._last_trigger_time[field.name]
        if timestep - pcts < fp.stride_timestep:
            return False
        if t - pct < fp.stride_time:
            return False

        # Accept!
        return True

    def _should_finalize_at_this_time(self, field, t, timestep):
        "Check if field is completed"
        # If we later wish to move configuration of field compute frequencies to NSPostProcessor,
        # it's easy to swap here with e.g. fp = self._field_params[field.name]
        fp = field.params
        
        # Should never be finalized
        #if not fp["finalize"]:
        #    return False
        
        # Finalize if that is default for field
        # FIXME: This shouldn't be necessary. Required today to return a "running" timeintegral,
        # while also making sure that end-points are included.
        if not field.__class__.default_params()["finalize"] and not fp["finalize"]:
            return False
        
        # Already finalized
        #if field.name in self._finalized:
        #    return False

        e = fp.end_timestep
        if e <= timestep:
            return True

        # Limit by time interval
        e = fp.end_time
        eps = 1e-10
        if e-eps < t:
            return True
        
        # Not completed
        return False

    def _rollback_plan(self):
        "Roll plan one timestep and countdown how long to keep stuff."
        tss = sorted(self._plan.keys())
        new_plan = defaultdict(lambda: defaultdict(int))

        # Countdown time to keep each data item and only keep what we still need
        for ts in tss:
            for name, ttk in self._plan[ts].items():
                if ttk > 0:
                    new_plan[ts-1][name] = ttk - 1
        self._plan = new_plan

    def update(self, fields, full_dependencies, dependencies, t, timestep):
        """Update plan for new timestep. Will trigger fields that are to be
        computed at this time to plan. Will estimate timestep (delta t), to
        prepare for computations at coming timestep by triggering dependencies
        to be computed at the required time.
        """
        #"Update plan for new timestep."
        # ttk = timesteps to keep
        #self._plan[-1][name] = ttk # How long to cache what's already computed
        #self._plan[0][name] = ttk  # What to compute now and how long to cache it
        #self._plan[1][name] = ttk  # What to compute in future and how long to cache it

        self._rollback_plan()

        # TODO: Allow for varying timestep
        #min_dt_factor = 0.5
        #max_dt_factor = 2.0
        #dt = self._problem.params.dt
        if self._t_prev == None:
            dt = self._initial_dt
        else:
            dt = t-self._t_prev
        self._t_prev = t
        
        finalize_fields = []
        
        # Loop over all fields that and trigger for computation
        #for name, field in self._fields.iteritems():
        for name, field in fields.iteritems():
            full_deps = full_dependencies[name]
            offset = abs(min([ts for depname, ts in full_deps]+[0]))
                        
            # Check if field should be computed in this or the next timesteps,
            # and plan dependency computations accordingly
            # TODO: Allow for varying timestep
            for i in range(offset+1):
                ti = t+i*dt
                if self._should_compute_at_this_time(field, ti, timestep+i):
                    if i == 0:
                        # Store compute trigger times to keep track of compute intervals
                        self._last_trigger_time[field.name] = (t, timestep)
                        
                    self._insert_deps_in_plan(dependencies, name, i)
                    
                    oldttk = self._plan[0].get(name, 0)
                    ttk = max(oldttk, i)
                    self._plan[i][name] = ttk
                    
            #field = self._fields[name]    
            if self._should_finalize_at_this_time(field, t, timestep):
                finalize_fields.append(name)
        
        return self._plan, finalize_fields, self._last_trigger_time

    def _insert_deps_in_plan(self, dependencies, name, offset):
        "Insert dependencies recursively in plan"
        deps = dependencies[name]
        for depname, ts in deps:
            # Find time-to-keep (FIXME: Is this correct? Optimal?)
            # What about: ttk = max(oldttk, offset-ts)
            oldttk = self._plan[ts+offset].get(depname, 0)
            ttk = max(oldttk, offset-min([_ts for _depname, _ts in deps if _depname==depname])+ts)
            

            # Insert in plan if able to compute
            if offset+ts >= 0:
                self._plan[offset+ts][depname] = ttk
            
                new_offset = offset+ts
                self._insert_deps_in_plan(dependencies, depname, new_offset)
        