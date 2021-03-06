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
By replaying a problem, we mean using stored data from a simulation
to compute any requested additional data.

For usage of this refer to :ref:`Replay`.

"""
import os
import shelve
from operator import itemgetter
import copy

from cbcpost import Parameterized, ParamDict, PostProcessor
from cbcpost.utils import (cbc_print, Timer, Loadable, loadable_formats, create_function_from_metadata,
                           get_memory_usage)

from dolfin import MPI, mpi_comm_world

class MiniCallable():
    """Mini callable as a replacement for lambda-functions. """
    def __init__(self, value):
        self.value = value

    def __call__(self):
        """ Simply return self.value """
        return self.value

def have_necessary_deps(solution, pp, field):
    """Check if field have necessary dependencies within given solution
    and postprocessor."""
    if field in solution:
        return True

    deps = pp._dependencies[field]
    if len(deps) == 0:
        return False

    all_deps = []
    for dep in deps:
        all_deps.append(have_necessary_deps(solution, pp, dep[0]))
    return all(all_deps)

class Replay(Parameterized):
    """ Replay class for postprocessing exisiting solution data. """
    def __init__(self, postprocessor, params=None):
        Parameterized.__init__(self, params)
        self.postproc = postprocessor
        self._functions = {}

        self.timer = self.postproc._timer

    @classmethod
    def default_params(cls):
        """
        Default parameters are:

        +------------------------+-----------------------+--------------------------------------------------------------+
        |Key                     | Default value         |  Description                                                 |
        +========================+=======================+==============================================================+
        | check_memory_frequency | 0                     | Frequency to report memory usage                             |
        +------------------------+-----------------------+--------------------------------------------------------------+

        """

        params = ParamDict(
            check_memory_frequency=0,
        )
        return params

    def _fetch_history(self):
        "Read playlog and create Loadables for all data that are available to read"
        playlog = self.postproc.get_playlog('r')
        data = {}
        replay_solutions = {}
        for key, value in playlog.items():
            replay_solutions[int(key)] = {"t": value["t"]}
            data[int(key)] = value

        from cbcpost import ConstantField
        constant_fields = [f for f in self.postproc._fields.values() if isinstance(f, ConstantField)]

        playlog.close()
        metadata_files = {}
        for timestep in sorted(data.keys()):
            for f in constant_fields:
                replay_solutions[timestep][f.name] = MiniCallable(f.compute(None))

            if "fields" not in data[timestep]:
                continue

            for fieldname, fieldnamedata in data[timestep]["fields"].items():
                if not any([saveformat in loadable_formats for saveformat in fieldnamedata["save_as"]]):
                    continue

                if fieldname not in metadata_files:
                    metadata_files[fieldname] = shelve.open(os.path.join(self.postproc.get_savedir(fieldname), "metadata.db"), 'r')

                if 'hdf5' in fieldnamedata["save_as"]:
                    function = self._get_function(fieldname, metadata_files[fieldname], 'hdf5')
                    filename = os.path.join(self.postproc.get_savedir(fieldname), fieldname+'.hdf5')
                    saveformat = "hdf5"
                elif 'xml' in fieldnamedata["save_as"]:
                    function = self._get_function(fieldname, metadata_files[fieldname], 'xml')
                    filename = os.path.join(self.postproc.get_savedir(fieldname), fieldname+str(timestep)+'.xml')
                    saveformat = "xml"
                elif 'xml.gz' in fieldnamedata["save_as"]:
                    function = self._get_function(fieldname, metadata_files[fieldname], 'xml.gz')
                    filename = os.path.join(self.postproc.get_savedir(fieldname), fieldname+'.xml.gz')
                    saveformat = "xml.gz"
                elif "shelve" in fieldnamedata["save_as"]:
                    function = None
                    filename = os.path.join(self.postproc.get_savedir(fieldname), fieldname+'.db')
                    saveformat = "shelve"
                else:
                    raise RuntimeError("Unable to find readable saveformat for field %s" %fieldname)
                replay_solutions[timestep][fieldname] = Loadable(filename, fieldname, timestep, data[timestep]["t"], saveformat, function)

        # Close all metadata files
        [f.close() for f in metadata_files.values()]

        return replay_solutions

    def _check_field_coverage(self, plan, fieldname):
        "Find which timesteps fieldname can be computed at"
        timesteps = []
        for ts in plan.keys():
            if self._recursive_dependency_check(plan, ts, fieldname):
                timesteps.append(ts)

        return timesteps

    def _recursive_dependency_check(self, plan, key, fieldname):
        "Check if field or dependencies exist in plan"
        if key not in plan:
            return False
        if fieldname == "t":
            return True

        # Return True if data present in plan as a readable data format
        if plan[key].get(fieldname):
            return True

        # If no dependencies, or if not all dependencies exist in plan, return False
        dependencies = self.postproc._dependencies[fieldname]
        if len(dependencies) == 0:
            return False
        else:
            for dep_field, dep_time in dependencies:
                if dep_time != 0:
                    continue
                if not self._recursive_dependency_check(plan, key, dep_field):
                    return False
            return True


    def _get_function(self, fieldname, metadata, saveformat):
        if fieldname not in self._functions:
            self._functions[fieldname] = create_function_from_metadata(self.postproc, fieldname, metadata, saveformat)
        return self._functions[fieldname]

    def backup_playlog(self):
        "Create a backup of the playlog"
        if MPI.rank(mpi_comm_world()) == 0:
            casedir = self.postproc.get_casedir()
            playlog_file = os.path.join(casedir, "play.db")
            i = 0
            backup_file = playlog_file+".bak"+str(i)

            while os.path.isfile(backup_file):
                i += 1
                backup_file = playlog_file+".bak"+str(i)
            os.system("cp %s %s" %(playlog_file, backup_file))
        MPI.barrier(mpi_comm_world())

    def revert_casedir(self):
        "Use backup playlog to remove all (newly introduced) fields."
        if MPI.rank(mpi_comm_world()) == 0:
            current_playlog = dict(self.postproc.get_playlog('r'))

            casedir = self.postproc.get_casedir()
            playlog_file = os.path.join(casedir, "play.db")
            i = 0
            backup_file = playlog_file+".bak"+str(i)
            while os.path.isfile(backup_file):
                i += 1
                backup_file = playlog_file+".bak"+str(i)
            i -= 1
            backup_file = playlog_file+".bak"+str(i)

            os.system("cp %s %s" %(backup_file, playlog_file))
            os.system("rm %s" %backup_file)

            backup_playlog = self.postproc.get_playlog('r')
            assert set(current_playlog.keys()) == set(backup_playlog.keys())

            current_fields = set()
            backup_fields = set()

            keys = [k for k in current_playlog.keys() if k.isdigit()]
            for k in keys:
                current_fields.update(current_playlog[k].get("fields", dict()).keys())
                backup_fields.update(backup_playlog[k].get("fields",dict()).keys())

            fields_to_remove = current_fields-backup_fields

            for field in fields_to_remove:
                os.system("rm -r %s/%s" %(casedir,field))


    def replay(self):
        "Replay problem with given postprocessor."
        # Backup play log
        self.backup_playlog()

        # Set up for replay
        replay_plan = self._fetch_history()
        postprocessors = []
        for fieldname, field in self.postproc._fields.items():
            if not (field.params.save
                    or field.params.plot):
                continue

            # Check timesteps covered by current field
            keys = self._check_field_coverage(replay_plan, fieldname)

            # Get the time dependency for the field
            t_dep = min([dep[1] for dep in self.postproc._dependencies[fieldname]]+[0])

            dep_fields = []
            for dep in self.postproc._full_dependencies[fieldname]:
                if dep[0] in ["t", "timestep"]:
                    continue

                if dep[0] in dep_fields:
                    continue

                # Copy dependency and set save/plot to False. If dependency should be
                # plotted/saved, this field will be added separately.
                dependency = self.postproc._fields[dep[0]]
                dependency = copy.copy(dependency)
                dependency.params.save = False
                dependency.params.plot = False
                dependency.params.safe = False

                dep_fields.append(dependency)

            added_to_postprocessor = False
            for i, (ppkeys, ppt_dep, pp) in enumerate(postprocessors):
                if t_dep == ppt_dep and set(keys) == set(ppkeys):
                    pp.add_fields(dep_fields, exists_reaction="ignore")
                    pp.add_field(field, exists_reaction="replace")

                    added_to_postprocessor = True
                    break
                else:
                    continue

            # Create new postprocessor if no suitable postprocessor found
            if not added_to_postprocessor:
                pp = PostProcessor(self.postproc.params, self.postproc._timer)
                pp.add_fields(dep_fields, exists_reaction="ignore")
                pp.add_field(field, exists_reaction="replace")

                postprocessors.append([keys, t_dep, pp])

        postprocessors = sorted(postprocessors, key=itemgetter(1), reverse=True)

        t_independent_fields = []
        for fieldname in self.postproc._fields:
            if self.postproc._full_dependencies[fieldname] == []:
                t_independent_fields.append(fieldname)
            elif min(t for dep,t in self.postproc._full_dependencies[fieldname]) == 0:
                t_independent_fields.append(fieldname)

        # Run replay
        sorted_keys = sorted(replay_plan.keys())
        N = max(sorted_keys)
        for timestep in sorted_keys:
            cbc_print("Processing timestep %d of %d. %.3f%% complete." %(timestep, N, 100.0*(timestep)/N))

            # Load solution at this timestep (all available fields)
            solution = replay_plan[timestep]
            t = solution.pop("t")

            # Cycle through postprocessors and update if required
            for ppkeys, ppt_dep, pp in postprocessors:
                if timestep in ppkeys:
                    # Add dummy solutions to avoid error when handling dependencies
                    # We know this should work, because it has already been established that
                    # the fields to be computed at this timestep can be computed from stored
                    # solutions.
                    for field in pp._sorted_fields_keys:
                        for dep in reversed(pp._dependencies[field]):
                            if not have_necessary_deps(solution, pp, dep[0]):
                                solution[dep[0]] = lambda: None
                    pp.update_all(solution, t, timestep)

                    # Clear None-objects from solution
                    [solution.pop(k) for k in solution.keys() if not solution[k]]

                    # Update solution to avoid re-computing data
                    for fieldname,value in pp._cache[0].items():
                        if fieldname in t_independent_fields:
                            value = pp._cache[0][fieldname]
                            #solution[fieldname] = lambda value=value: value # Memory leak!
                            solution[fieldname] = MiniCallable(value)


            self.timer.increment()
            if self.params.check_memory_frequency != 0 and timestep%self.params.check_memory_frequency==0:
                cbc_print('Memory usage is: %s' % MPI.sum(mpi_comm_world(), get_memory_usage()))

            # Clean up solution: Required to avoid memory leak for some reason...
            for f, v in solution.items():
                if isinstance(v, MiniCallable):
                    v.value = None
                    del v
                    solution.pop(f)

        for ppkeys, ppt_dep, pp in postprocessors:
            pp.finalize_all()

