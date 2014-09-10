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

import pickle
import os
import shelve

from cbcpost import Parameterized, ParamDict, PostProcessor, SpacePool
from cbcflow.utils.common import cbcflow_print, Timer

from dolfin import HDF5File, Mesh, Function, FunctionSpace, VectorFunctionSpace, TensorFunctionSpace, BoundaryMesh

fetchable_formats = ["hdf5", "xml", "xml.gz", "shelve"]

def print_replay_plan(plan):
    for timestep in sorted(plan.keys()):
        print timestep, plan[timestep].keys()


def have_necessary_deps(solution, pp, field):   
    if field in solution:
        return True
    
    deps = pp._dependencies[field]
    if len(deps) == 0:
        return False
    
    all_deps = []
    for dep in deps:
        all_deps.append(have_necessary_deps(solution, pp, dep[0]))
    return all(all_deps)


class Loadable():
    def __init__(self, filename, fieldname, timestep, time, saveformat, function):
        self.filename = filename
        self.fieldname = fieldname
        self.timestep = timestep
        self.time = time
        self.saveformat = saveformat
        self.function = function
        
        assert self.saveformat in fetchable_formats
        
    def __call__(self):
        if self.saveformat == 'hdf5':
            hdf5file = HDF5File(self.filename, 'r')
            hdf5file.read(self.function, self.fieldname+str(self.timestep))
            del hdf5file
            return self.function
        elif self.saveformat in ["xml", "xml.gz"]:
            V = self.function.function_space()
            self.function.assign(Function(V, self.filename))
            return self.function
        elif self.saveformat == "shelve":
            shelvefile = shelve.open(self.filename)
            return shelvefile[str(timestep)]
    
class Replay(Parameterized):
    """ Replay class for postprocessing exisiting solution data. """
    def __init__(self, postprocessor, params=None):
        Parameterized.__init__(self, params)
        self.postproc = postprocessor
        self._functions = {}
        
        self.timer = Timer(self.params.timer_frequency)
        self.timer._N = 0
    
    @classmethod
    def default_params(cls):
        params = ParamDict(
            timer_frequency=0,
        )
        return params

    def _fetch_history(self):
        casedir = self.postproc.get_casedir()
        assert(os.path.isfile(os.path.join(casedir, "play.db")))
       
        # Read play.shelve
        play = shelve.open(os.path.join(casedir, "play.db"))
        data = {}
        replay_solutions = {}
        for key, value in play.items():
            replay_solutions[int(key)] = {"t": value["t"]}
            data[int(key)] = value
        
        metadata_files = {}
        for timestep in sorted(data.keys()):
            if "fields" not in data[timestep]:
                continue
            
            for fieldname, fieldnamedata in data[timestep]["fields"].items():
                if not any([saveformat in fetchable_formats for saveformat in fieldnamedata["save_as"]]):
                    continue
                
                if fieldname not in metadata_files:
                    metadata_files[fieldname] = shelve.open(os.path.join(casedir, fieldname, "metadata.db"))
                
                if 'hdf5' in fieldnamedata["save_as"]:
                    function = self._get_function(fieldname, metadata_files[fieldname], 'hdf5')
                    filename = os.path.join(self.postproc.get_savedir(fieldname), fieldname+'.hdf5')
                elif 'xml' in fieldnamedata["save_as"]:
                    function = self._get_function(fieldname, metadata_files[fieldname], 'xml')
                    filename = os.path.join(self.postproc.get_savedir(fieldname), fieldname+str(timestep)+'.hdf5')
                elif 'xml.gz' in fieldnamedata["save_as"]:
                    function = self._get_function(fieldname, metadata_files[fieldname], 'xml.gz')
                    filename = os.path.join(self.postproc.get_savedir(fieldname), fieldname+'.db')
                else:
                    function = None
                
                replay_solutions[timestep][fieldname] = Loadable(filename, fieldname, timestep, data[timestep]["t"], saveformat, function)
                
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
            checks = []
            for dep_field, dep_time in dependencies:
                if dep_time != 0:
                    continue
                checks.append(self._recursive_dependency_check(plan, key, dep_field))
            return all(checks)
     
    def _get_mesh(self):
        if not hasattr(self, "_mesh"):       
            # Read mesh
            meshfilename = os.path.join(self.postproc.get_casedir(), "mesh.hdf5")
            assert os.path.isfile(meshfilename), "Unable to find mesh file!"
            meshfile = HDF5File(meshfilename, 'r')
            self._mesh = Mesh()
            meshfile.read(self._mesh, "Mesh")
            del meshfile

        return self._mesh
    
    def _get_boundarymesh(self):
        if not hasattr(self, "_boundarymesh"):
            self._boundarymesh = BoundaryMesh(self._get_mesh(), 'exterior')

        return self._boundarymesh

    def _create_function_from_metadata(self, fieldname, metadata, saveformat):
        assert metadata['type'] == 'Function'
        
        # Load mesh
        if saveformat == 'hdf5':    
            mesh = Mesh()
            hdf5file = HDF5File(os.path.join(self.postproc.get_casedir(),fieldname, fieldname+'.hdf5'), 'r')
            hdf5file.read(mesh, "Mesh")
            del hdf5file
        elif saveformat == 'xml' or saveformat == 'xml.gz':
            mesh = Mesh(os.path.join(self.postproc.get_casedir(), fieldname, "mesh."+saveformat))
        
        shape = eval(metadata["element_value_shape"])
        degree = eval(metadata["element_degree"])
        family = eval(metadata["element_family"])
        
        # Get space from existing function spaces if mesh is the same
        spaces = SpacePool(mesh)
        space = spaces.get_custom_space(family, degree, shape)
        """
        # TODO: Verify that this check is good enough
        if mesh.hash() != self._get_mesh().hash():
            if mesh.hash() == self._get_boundarymesh().hash():
                mesh = self._get_boundarymesh()
            rank = len(shape)
            if rank == 0:
                space = FunctionSpace(mesh, family, degree)
            elif rank == 1:
                space = VectorFunctionSpace(mesh, family, degree)
            elif rank == 2:
                space = TensorFunctionSpace(mesh, family, degree)
        else:
            del mesh
            space = self._get_spaces().get_space(degree, len(shape), family)
        """
        return Function(space, name=fieldname)
    """
    def _get_all_params(self):
        paramfile = open(os.path.join(self.postproc.get_casedir(), "params.pickle"), 'rb')
        return pickle.load(paramfile)
    """
    def _get_function(self, fieldname, metadata, saveformat):
        if fieldname not in self._functions:
            self._functions[fieldname] = self._create_function_from_metadata(fieldname, metadata, saveformat)
        return self._functions[fieldname]
       
    def replay(self):
        "Replay problem with given postprocessor."
        # Set up for replay
        replay_plan = self._fetch_history()
        postprocessors = []
        for fieldname, field in self.postproc._fields.items():
            if not (field.params.save
                    or field.params.plot
                    or field.params.callback):
                continue
            
            keys = self._check_field_coverage(replay_plan, fieldname)
            # Check timesteps covered by current field
            keys = self._check_field_coverage(replay_plan, fieldname)
            print fieldname#, keys
            
            # Get the time dependency for the field
            t_dep = min([dep[1] for dep in self.postproc._dependencies[fieldname]]+[0])
            
            dep_fields = []
            for dep in self.postproc._full_dependencies[fieldname]:
                if dep[0] in ["t", "timestep"]:
                    continue
                
                if dep[0] in dep_fields:
                    continue
                
                dep_fields.append(self.postproc._fields[dep[0]])
            fields = dep_fields + [field]
            
            added_to_postprocessor = False
            for i, (ppkeys, ppt_dep, pp) in enumerate(postprocessors):
                if t_dep == 0 and set(keys).issubset(set(ppkeys)):
                    # TODO: Check this extend
                    ppkeys.extend(keys)
                    pp.add_fields(fields)
                    added_to_postprocessor = True
                    break
                elif t_dep == ppt_dep and keys == ppkeys:
                    pp.add_fields(fields)
                    added_to_postprocessor = True
                    break
                else:
                    continue
                    
            # Create new postprocessor if no suitable postprocessor found
            if not added_to_postprocessor:
                pp = PostProcessor({"casedir": self.postproc.get_casedir()})
                pp._timer = self.timer
                #dep_fields = list(set([self.postproc._fields[dep[0]] for dep in self.postproc._full_dependencies[fieldname] if dep[0] not in ["t", "timestep"]]))
                fields = dep_fields + [field]
                #import ipdb; ipdb.set_trace()
                pp.add_fields(fields)
                postprocessors.append([keys, t_dep, pp])
            
            
        """
        for fieldname in self.postproc._sorted_fields_keys:
            field = self.postproc._fields[fieldname]
            if not field.params.save:
                continue

            # Check timesteps covered by current field
            keys = self._check_field_coverage(replay_plan, fieldname)
            print fieldname#, keys
            
            # Get the time dependency for the field
            t_dep = min([dep[1] for dep in self.postproc._dependencies[fieldname]]+[0])

            # Append field to correct postprocessor
            # TODO: Determine what the best way to do this is
            added_to_postprocessor = False
            for i, (ppkeys, ppt_dep, pp) in enumerate(postprocessors):
                if t_dep == 0 and set(keys).issubset(set(ppkeys)):
                    # TODO: Check this extend
                    ppkeys.extend(keys)
                    pp.add_field(field)
                    added_to_postprocessor = True
                    break
                elif t_dep == ppt_dep and keys == ppkeys:
                    pp.add_field(field)
                    added_to_postprocessor = True
                    break
                else:
                    continue
                    
            # Create new postprocessor if no suitable postprocessor found
            if not added_to_postprocessor:
                pp = PostProcessor({"casedir": self.postproc.get_casedir()})
                dep_fields = list(set([self.postproc._fields[dep[0]] for dep in self.postproc._full_dependencies[fieldname] if dep[0] not in ["t", "timestep"]]))
                fields = dep_fields + [field]
                #import ipdb; ipdb.set_trace()
                pp.add_fields(fields)
                postprocessors.append([keys, t_dep, pp])
        """
        # Run replay
        for timestep in sorted(replay_plan.keys()):
            cbcflow_print("Processing timestep %d of %d. %.3f%% complete." %(timestep, max(replay_plan.keys()), 100.0*(timestep)/(max(replay_plan.keys()))))

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
                    #pp.update_all(solution, t, timestep, self._get_spaces(), problem)
                    pp.update_all(solution, t, timestep)
                    
                    # Clear None-objects from solution
                    [solution.pop(k) for k in solution.keys() if not solution[k]]

                    # Update solution to avoid re-reading data
                    solution = pp._solution
                    
            self.timer.increment()
        
        for ppkeys, ppt_dep, pp in postprocessors:
            pp.finalize_all()

