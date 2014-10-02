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
"""Common functionality and interface for all Field-implementations."""
from os.path import join

from dolfin import Function, TestFunction, assemble, inner, dx, project, HDF5File, error
import shelve
from cbcpost import ParamDict, Parameterized

class Field(Parameterized):
    """ Base class for all fields.
    
    :param name: Specify name for field. If default, a name will be created based on class-name.
    :param label: Specify a label. The label will be added to the name, if name is default.
    
    """
    def __init__(self, params=None, name="default", label=None):
        Parameterized.__init__(self, params)
        if label:
            self.label = str(label)
        else:
            self.label = None
        self._name = name
        
    # --- Parameters

    @classmethod
    def default_save_as(cls):
        """ Specify default save formats for field. Default is *determined_by_data*."""
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
        """Return name of field, by default this is *classname-label*,
        but can be overloaded in subclass.
        """
        if self._name == "default":
            n = self.__class__.__name__
            if self.label: n += "-"+self.label
        else:
            n = self._name
        return n

    # --- Main interface
    def add_fields(self):
        "Specify any specific fields used in this field. Could be e.g. definite integrals."
        return []

    def before_first_compute(self, get):
        "Called before first call to compute."
        pass

    def after_last_compute(self, get):
        "Called after last call to compute."
        return "N/A"

    def compute(self, get):
        "Called each time the quantity should be computed."
        raise NotImplementedError("A Field must implement the compute function!")

    # --- Helper functions

    def expr2function(self, expr, function):
        """ Convert an expression into a function. How this is done is
        determined by the parameters (assemble, project or interpolate).
        """
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
