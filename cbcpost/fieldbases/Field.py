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

from os.path import join

from dolfin import Function, TestFunction, assemble, inner, dx, project, HDF5File, error
import shelve
from cbcpost import ParamDict, Parameterized

class Field(Parameterized):
    def __init__(self, params=None, name="default", label=None):
        Parameterized.__init__(self, params)
        if label:
            self.label = str(label)
        else:
            self.label = None
        
        self._name = self.__class__.__name__ if name == "default" else name

    # --- Parameters

    @classmethod
    def default_save_as(cls):
        return "determined by data"

    @classmethod
    def default_params(cls):
        params = ParamDict(
            # Configure direct compute requests through timestep counting
            start_timestep = -1e16,
            end_timestep = 1e16,
            stride_timestep = 1,

            # Configure direct compute requests through physical time intervals
            start_time = -1e16,
            end_time = 1e16,
            stride_time = 1e-16,

            # Trigger action after each direct compute request
            plot = False,
            save = False,
            callback = False,

            # Configure computing
            project = False, # This is the safest approach
            assemble = True, # This is faster but only works for for DG0
            interpolate = False, # This will be the best when properly implemented in fenics

            # Configure saving
            save_as = cls.default_save_as(),

            # Configure plotting
            plot_args={},
            
            # Solution switch
            is_solution = False,
            
            # Finalize field?
            finalize = False,
            )
        return params

    @property
    def name(self):
        "Return name of field, by default the classname but can be overloaded in subclass."
        n = self._name
        if self.label: n += "-"+self.label
        
        return n

    # --- Main interface
    def add_fields(self):
        "Specify any specific fields used in this field. Could be e.g. definite integrals."
        return []

    def before_first_compute(self, get):
        "Called prior to the simulation timeloop."
        pass

    def after_last_compute(self, get):
        "Called after the simulation timeloop."
        return "N/A"

    def compute(self, get):
        "Called each time the quantity should be computed."
        raise NotImplementedError("A Field must implement the compute function!")
    """
    def convert(self, pp, spaces, problem):
        
        # Load data from disk (this is used in replay functionality)
        # The structure of the dict pp._solution[self.name] is determined in nsreplay.py
        if isinstance(pp._solution[self.name], dict):
            timestep = get("timestep")
            saveformat = pp._solution[self.name]["format"]
            if saveformat == 'hdf5':
                hdf5filepath = join(get_casedir(), self.name, self.name+".hdf5")
                hdf5file = HDF5File(hdf5filepath, 'r')
                dataset = self.name+str(timestep)
                hdf5file.read(pp._solution[self.name]["function"], dataset)
                pp._solution[self.name] = pp._solution[self.name]["function"]
            elif saveformat in ["xml", "xml.gz"]:
                xmlfilename = self.name+str(timestep)+"."+saveformat
                xmlfilepath = join(get_casedir(), self.name, xmlfilename)
                function = pp._solution[self.name]["function"]
                function.assign(Function(function.function_space(), xmlfilepath))
                pp._solution[self.name] = pp._solution[self.name]["function"]
            elif saveformat == "shelve":
                shelvefilepath = join(get_casedir(), self.name, self.name+".db")
                shelvefile = shelve.open(shelvefilepath)
                pp._solution[self.name] = shelvefile[str(timestep)]

        return pp._solution[self.name]
    """
    # --- Helper functions

    def expr2function(self, expr, function):

        space = function.function_space()

        if self.params.assemble:
            # Compute average values of expr for each cell and place in a DG0 space

            # TODO: Get space from pool
            #shape = expr.shape()
            #space = pp.space_pool.get_custom_space("DG", 0, shape)
            #target = pp.function_pool.borrow_function(space)

            test = TestFunction(space)
            scale = 1.0 / space.mesh().ufl_cell().volume
            assemble(scale*inner(expr, test)*dx(), tensor=function.vector())
            return function

        elif self.params.project:
            # TODO: Avoid superfluous function creation by allowing project(expr, function=function) or function.project(expr)
            function.assign(project(expr, space))
            return function

        elif self.params.interpolate:
            # TODO: Need interpolation with code generated from expr, waiting for uflacs work.
            function.interpolate(expr) # Currently only works if expr is a single Function
            return function

        else:
            error("No action selected, need to choose either assemble, project or interpolate.")
