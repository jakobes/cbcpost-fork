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
The main module of cbcpost, which, for basic functionality, is the only
interface necessary for the user.
"""
from dolfin import Function, MPI, mpi_comm_world

from cbcpost import ParamDict, Parameterized
from cbcpost.plotter import Plotter
from cbcpost.planner import Planner
from cbcpost.saver import Saver
from cbcpost.fieldbases import Field

from cbcpost.utils import Timer, cbc_log, cbc_warning, strip_code

import inspect, re
from collections import defaultdict


class DependencyException(Exception):
    "Common Exception-class to handle all exceptions related to dependency handling"
    def __init__(self, fieldname=None, dependency=None, timestep=None, original_exception_msg=None):
        message = []
        if fieldname:
            message += ["Dependency/dependencies not found for field %s." % fieldname]
        if dependency:
            message += ["Dependency %s not functioning." % dependency]
        if timestep:
            message += ["Relative timestep is %d. Are you trying to calculate \
                        time-derivatives at t=0 without setting a correct \
                        initial dt?" % (timestep)]
        if original_exception_msg:
            message += ["\nOriginal exception was: " + original_exception_msg]
        message = ' '.join(message)
        Exception.__init__(self, message)

# Fields available through get(name) even though they have no Field class
builtin_fields = ("t", "timestep")

def find_dependencies(field):
    "Read dependencies from source code in field.compute function"
    
    # Get source of compute and after_last_compute
    s = inspect.getsource(field.compute)
    s += inspect.getsource(field.after_last_compute)
    s = strip_code(s) # Removes all comments, empty lines etc.

    # Remove comment blocks
    s = s.split("'''")
    s = s[0::2]
    s = ''.join(s)
    s = s.split('"""')
    s = s[0::2]
    s = ''.join(s)
    s = strip_code(s)

    # Get argument names for the compute function
    args = inspect.getargspec(field.compute)[0]
    self_arg = args[0]
    get_arg = args[1]

    # Read the code for dependencies
    deps = []
    deps_raw = re.findall(get_arg+"\((.+)\)", s)
    for dep in deps_raw:
        # Split into arguments (name, timestep)
        dep = dep.split(',')

        # Append default 0 if dependent timestep not specified
        if len(dep) == 1:
            dep.append(0)

        # Convert timestep to int
        dep[1] = int(dep[1])

        # Get field name from string literal or through string variable
        dep[0] = dep[0].strip(' ').replace('"', "'")
        if "'" in dep[0]:
            # If get('Velocity')
            dep[0] = dep[0].replace("'","")
        else:
            # If get(self.somevariable), get the string hiding at self.somevariable
            dep[0] = eval(dep[0].replace(self_arg, "field", 1))

            # TODO: Test alternative solution without eval (a matter of principle) and with better check:
            #s, n = dep[0].split(".")
            #assert s == self_arg, "Only support accessing strings through self."
            #dep[0] = getattr(field, n)

            # TODO: Possible to get variable in other cases through more introspection?
            #       Probably not necessary, just curious.

        # Append to dependencies
        deps.append(tuple(dep))
    
    # Make unique (can happen that deps are repeated in rare cases)
    return sorted(set(deps))


class PostProcessor(Parameterized):
    """
    All basic user interface is gathered here.
    """
    def __init__(self, params=None, timer=None):
        Parameterized.__init__(self, params)
        if isinstance(timer, Timer):
            self._timer = timer
        elif timer:
            self._timer = Timer(1)
        else:
            self._timer = Timer()
            
        self._extrapolate = self.params.extrapolate

        # Storage of actual fields
        self._fields = {}

        # Representations of field dependencies
        self._sorted_fields_keys = [] # Topological ordering of field names
        self._dependencies = {} # Direct dependencies dep[name] = ((depname0, ts0), (depname1, ts1), ...)
        self._full_dependencies = {} # Indirect dependencies included
        for depname in builtin_fields:
            self._dependencies[depname] = []
            self._full_dependencies[depname] = []

        # Create instances required for plotting, saving and planning
        #self._reverse_dependencies = {} # TODO: Need this?
        self._plotter = Plotter(self._timer)
        self._saver = Saver(self._timer, self.params.casedir)
        self._planner = Planner(self._timer, self.params.initial_dt)
        
        # Plan of what to compute now and in near future
        self._plan = defaultdict(lambda: defaultdict(int))
        
        # Cache of computed values needed for planned computations
        self._cache = defaultdict(dict)
        
        # Solution provided to update_all
        self._solution = dict()
        
        # Keep track of which fields have been finalized
        self._finalized = {}
        
        # Keep track of how many times update_all has been called
        self._update_all_count = 0
        
        # Keep track of how many times .get has called each field.compute, for administration:
        self._compute_counts = defaultdict(int) # Actually only used for triggering "before_first_compute"
        
        if self.params.clean_casedir: self._saver._clean_casedir()
        
        """
        # Callback to be called with fields where the 'callback' action is enabled
        # Signature: ppcallback(field, data, t, timestep)
        self._callback = None

        # Hack to make these objects available throughout during update... Move these to a struct?
        self._solution = None
        
        self._timer = Timer()
        """
    
    @classmethod
    def default_params(cls):
        """
        Default parameters are:
        

        ===============       ==============   ============================
        Key                   Default value    Description
        ===============       ==============   ============================
        casedir               '.'              Case directory - relative
                                               path to use for saving
        enable_timer          False            Enable timer
        extrapolate           True             Constant extrapolation of
                                               fields prior to first
                                               update call
        initial_dt            1e-5             Initial timestep. Only used
                                               in planning algorithm
                                               at first update call.
        clean_casedir         False            Clean out case directory
                                               prior to update.
        ===============       ==============   ============================
        
        
        Trying with a different table format:
        
        +----------------------+-----------------------+--------------------------------------------------------------+
        |Key                   | Default value         |  Description                                                 |
        +======================+=======================+==============================================================+
        | casedir              | '.'                   | Case directory - relative path to use for saving             |
        +----------------------+-----------------------+--------------------------------------------------------------+
        | enable_timer         | False                 | Enable timer                                                 |
        +----------------------+-----------------------+--------------------------------------------------------------+
        | extrapolate          | True                  | Constant extrapolation of fields prior to first              |
        |                      |                       | | update call                                                |
        +----------------------+-----------------------+--------------------------------------------------------------+
        | initial_dt           | 1e-5                  | Initial timestep. Only used in planning algorithm at first   |
        |                      |                       | | update call.                                               |
        +----------------------+-----------------------+--------------------------------------------------------------+
        | clean_casedir        | False                 | Clean out case directory prior to update.                    |
        +----------------------+-----------------------+--------------------------------------------------------------+
        
        """
        params = ParamDict(
            casedir=".",
            enable_timer=False,
            extrapolate=True,
            initial_dt=1e-5,
            clean_casedir=False,
            )
        return params
    
    def _insert_in_sorted_list(self, fieldname):
        # Topological ordering of all fields, so that all dependencies are taken care of

        # If already in list, assuming there's no change to dependencies
        if fieldname in self._sorted_fields_keys:
            return

        # Find largest index of dependencies in sorted list
        deps = [dep[0] for dep in self._dependencies[fieldname]
                if dep[0] not in builtin_fields]
        max_index = max([-1]+[self._sorted_fields_keys.index(dep) for dep in deps])

        # Insert item after all its dependencies
        self._sorted_fields_keys.insert(max_index+1, fieldname)

    def add_field(self, field):
        "Add field to postprocessor."
        assert isinstance(field, Field)
        
        # Note: If field already exists, replace anyway to overwrite params, this
        # typically happens when a fields has been created implicitly by dependencies.
        # This is a bit unsafe though, the user might add a field twice with different parameters...
        # Check that at least the same name is not used for different field classes:
        #assert field.name not in self._fields, "Field with name %s already been added to postprocessor." %field.name
        assert type(field) == type(self._fields.get(field.name,field))
        
        # Add fields explicitly specified by field
        self.add_fields(field.add_fields())

        # Analyze dependencies of field through source inspection
        deps = find_dependencies(field)
        for dep in deps:
            if dep[0] not in self._fields and dep[0] not in builtin_fields:
                raise DependencyException(fieldname=field.name, dependency=dep[0])

        # Build full dependency list
        full_deps = []
        existing_full_deps = set()
        for dep in deps:
            depname, ts = dep
            for fdep in self._full_dependencies[depname]:
                # Sort out correct (relative) timestep of dependency
                fdepname, fts = fdep
                fts += ts
                fdep = (fdepname, fts)
                if fdep not in existing_full_deps:
                    existing_full_deps.add(fdep)
                    full_deps.append(fdep)
            existing_full_deps.add(dep)
            full_deps.append(dep)
        
        # Add field to internal data structures
        self._fields[field.name] = field
        self._dependencies[field.name] = deps
        self._full_dependencies[field.name] = full_deps
        self._insert_in_sorted_list(field.name)
        
        cbc_log(20, "Added field: %s" %field.name)
        
        # Returning the field object is useful for testing
        return field

    def add_fields(self, fields):
        "Add several fields at once."
        return [self.add_field(field) for field in fields]
    
    def get(self, name, relative_timestep=0, compute=True, finalize=False):
        """Get the value of a named field at a particular relative_timestep.

        The relative_timestep is relative to now.
        Values are computed at first request and cached.
        """
        cbc_log(20, "Getting: %s, %d (compute=%s, finalize=%s)" %(name, relative_timestep, compute, finalize))
        
        # Check cache
        c = self._cache[relative_timestep]
        data = c.get(name, "N/A")
        
        # Check if field has been finalized, and if so,
        # return finalized value
        if name in self._finalized and data == "N/A":
            if compute:
                cbc_warning("Field %s has already been finalized. Will not call compute on field." %name)
            return self._finalized[name]
        
        # Are we attempting to get value from before update was started?
        # Use constant extrapolation if allowed.
        if abs(relative_timestep) > self._update_all_count and data == "N/A":
            if self._extrapolate:
                cbc_log(20, "Extrapolating %s from %d to %d" %(name, relative_timestep, -self._update_all_count))
                data = self.get(name, -self._update_all_count, compute, finalize)
                c[name] = data
            else:
                raise RuntimeError("Unable to get data from before update was started. \
                                   (%s, relative_timestep: %d, update_all_count: %d)"
                                   %(name, relative_timestep, self._update_all_count))
        # Cache miss?
        if data == "N/A":
            field = self._fields[name]
            if relative_timestep == 0:
                # Ensure before_first_compute is always called once initially
                if self._compute_counts[field.name] == 0:
                    init_data = field.before_first_compute(self.get)
                    self._timer.completed("PP: before first compute %s" %name)
                    if init_data != None:
                        cbc_warning("Did not expect a return value from \
                                    %s.before_first_compute." %field.__class__)

                # Compute value
                if name in self._solution:
                    data = self._solution[name]()
                    self._timer.completed("PP: call solution %s" %name)
                else:
                    if compute:
                        data = field.compute(self.get)
                        self._timer.completed("PP: compute %s" %name)
                    """
                    if finalize:
                        finalized_data = field.after_last_compute(self.get)
                        if finalized_data not in [None, "N/A"]:
                            data = finalized_data
                        self._finalized[name] = data
                        self._timer.completed("PP: finalize %s" %name)
                    """
                self._compute_counts[field.name] += 1

                # Copy functions to avoid storing references to the same function objects at each relative_timestep
                # NB! In other cases we assume that the fields return a new object for every compute!
                # Check first if we actually will cache this object by looking at 'time to keep' in the plan
                if self._plan[0][name] > 0:
                    if isinstance(data, Function):
                        # TODO: Use function pooling to avoid costly allocations?
                        data = Function(data)

                # Cache it!
                #c[name] = data
            else:
                # Cannot compute missing value from previous relative_timestep,
                # dependency handling must have failed
                raise DependencyException(name, relative_timestep=relative_timestep)

        if finalize:
            field = self._fields[name]
            finalized_data = field.after_last_compute(self.get)
            if finalized_data not in [None, "N/A"]:
                data = finalized_data
            self._finalized[name] = data
            self._timer.completed("PP: finalize %s" %name)

        c[name] = data
        return data

    def _execute_plan(self, t, timestep):
        "Check plan and compute fields in plan."
        
        # Initialize cache for current timestep
        assert not self._cache[0], "Not expecting cached computations at this \
                                    timestep, before plan execution!"
        self._cache[0] = {
            "t": t,
            "timestep": timestep,
            }

        # Loop over all planned field computations
        for name in self._sorted_fields_keys:
            if name in self._plan[0]:
                compute = True
            else:
                compute = False
            
            field = self._fields[name]    
            
            if name in self._finalize_plan:# and name not in self._finalized:
                finalize = True
            else:
                finalize = False
            
            # If neither finalize or compute triggers, continue
            if not (finalize or compute):
                continue
            
            # Execute computation through get call
            data = self.get(name, compute=compute, finalize=finalize)
    
    def _update_cache(self, t, timestep):
        "Update cache, remove what can be removed"
        new_cache = defaultdict(dict)
        # Loop over cache plans for each timestep
        for ts, plan in self._plan.iteritems():
            # Skip whats not computed yet
            if ts > 0:
                continue
            # Only keep what we have planned to cache
            for name, ttk in plan.iteritems():
                if ttk > 0:
                    # Cache should contain old cached values at ts<0
                    # and newly computed values at ts=0
                    data = self._cache[ts].get(name, "N/A")
                    assert data is not "N/A", "Missing cache data!"
                    # Insert data in new cache at position ts-1
                    new_cache[ts-1][name] = data
        self._cache = new_cache

    def update_all(self, solution, t, timestep):
        "Updates cache, plan, play log and executes plan."
        MPI.barrier(mpi_comm_world())

        # TODO: Better design solution to making these variables accessible the right places?
        self._solution = solution

        # Update play log
        self._saver._update_play_log(t, timestep)

        # Update cache to keep what's needed later according to plan, forget what we don't need
        self._update_cache(t, timestep)

        # Plan what we need to compute now and in near future based on action triggers and dependencies
        self._plan, self._finalize_plan, self._last_trigger_time = \
            self._planner.update(self._fields, self._full_dependencies, self._dependencies, t, timestep)
        self._timer.completed("PP: updated plan.")

        # Compute what's needed according to plan
        self._execute_plan(t, timestep)
        
        triggered_or_finalized = []
        for name in self._cache[0]:
            if (name in self._finalize_plan
                or self._last_trigger_time[name][1] == timestep):
                triggered_or_finalized.append(self._fields[name])
        
        self._saver.update(t, timestep, self._cache[0], triggered_or_finalized)
        self._plotter.update(t, timestep, self._cache[0], triggered_or_finalized)
        
        
        self._update_all_count += 1
        MPI.barrier(mpi_comm_world())

    def finalize_all(self):
        "Finalize all Fields after last timestep has been computed."
        finalized = []
        for name in self._sorted_fields_keys:
            field = self._fields[name]
            if field.params.finalize and name not in self._finalized:
                self.get(name, compute=False, finalize=True)
                finalized.append(field)
        
        t = self._cache[0].get("t", -1e16)
        timestep = self._cache[0].get("timestep", -1e16)
        
        self._saver.update(t, timestep, self._cache[0], finalized)
        self._plotter.update(t, timestep, self._cache[0], finalized)
        MPI.barrier(mpi_comm_world())


    def store_mesh(self, mesh, cell_domains=None, facet_domains=None):
        """Store mesh in casedir to mesh.hdf5 (dataset Mesh) in casedir.
        
        Forwarded to a Saver-instance.
        """
        self._saver.store_mesh(mesh, cell_domains, facet_domains)
        MPI.barrier(mpi_comm_world())

    def clean_casedir(self):
        """Cleans out all files produced by cbcpost in the current casedir.
        
        Forwarded to a Saver-instance.
        """
        self._saver._clean_casedir()

    def store_params(self, params):
        """Store parameters in casedir as params.pickle and params.txt.
        
        Forwarded to a Saver-instance.
        """
        self._saver.store_params(params)
        MPI.barrier(mpi_comm_world())
        
    def get_casedir(self):
        """Return case directory.
        
        Forwarded to a Saver-instance.
        """
        return self._saver.get_casedir()
    
    def get_savedir(self, fieldname):
        """Returns save directory for given field name
        
        Forwarded to a Saver-instance.
        """
        return self._saver.get_savedir(fieldname)
    
    def get_playlog(self):
        """
        Get play log from disk (which is stored as a shelve-object).
        
        Forwarded to a Saver-instance.
        """
        return self._saver._fetch_play_log()