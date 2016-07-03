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

from dolfin import TestFunction, assemble, inner, dx, project, error, CellVolume
from cbcpost import ParamDict, Parameterized


class Field(Parameterized):
    """ Base class for all fields.

    :param name: Specify name for field. If default, a name will be created based on class-name.
    :param label: Specify a label. The label will be added to the name, if name is default.

    """
    _recording = False
    _record = []
    
    def __init__(self, params=None, name="default", label=None):
        Parameterized.__init__(self, params)
        if label:
            self.label = str(label)
        else:
            self.label = None
        self._name = name

        if Field._recording:
            Field._record.append(self)

    # --- Parameters

    @classmethod
    def default_save_as(cls):
        """ Specify default save formats for field. Default is *determined_by_data*."""
        return "determined by data"

    @classmethod
    def default_params(cls):
        """
        Default params are:

        +----------------------+-----------------------+-------------------------------------------------------------------+
        |Key                   | Default value         |  Description                                                      |
        +======================+=======================+===================================================================+
        | start_timestep       | -1e16                 | Timestep to start computation                                     |
        +----------------------+-----------------------+-------------------------------------------------------------------+
        | end_timestep         | 1e16                  | Timestep to end computation                                       |
        +----------------------+-----------------------+-------------------------------------------------------------------+
        | stride_timestep      | 1                     | Number of steps between each computation                          |
        +----------------------+-----------------------+-------------------------------------------------------------------+
        | start_time           | -1e16                 | Time to start computation                                         |
        +----------------------+-----------------------+-------------------------------------------------------------------+
        | end_time             | 1e16                  | Time to end computation                                           |
        +----------------------+-----------------------+-------------------------------------------------------------------+
        | stride_time          | 1e-16                 | Time between each computation                                     |
        +----------------------+-----------------------+-------------------------------------------------------------------+
        | plot                 | False                 | Plot Field after a directly triggered computation                 |
        +----------------------+-----------------------+-------------------------------------------------------------------+
        | plot_args            | {}                    | Keyword arguments to pass to dolfin.plot.                         |
        +----------------------+-----------------------+-------------------------------------------------------------------+
        | safe                 | True                  | Trigger safe computation. This allows get-calls to this field     |
        |                      |                       | outside postprocessor. Set to False to rely on postprocessor      |
        |                      |                       | and improve efficiency.                                           |
        +----------------------+-----------------------+-------------------------------------------------------------------+
        | save                 | False                 | Save Field after a directly triggered computation                 |
        +----------------------+-----------------------+-------------------------------------------------------------------+
        | save_as              | 'determined by data'  | Format(s) to save in. Allowed save formats:                       |
        |                      |                       |                                                                   |
        |                      |                       | The default values are:                                           |
        |                      |                       |                                                                   |
        |                      |                       | - ['hdf5', 'xdmf'] if data is dolfin.Function                     |
        |                      |                       | - ['txt', 'shelve'] if data is float, int, list, tuple or dict    |
        |                      |                       |                                                                   |
        +----------------------+-----------------------+-------------------------------------------------------------------+
        | expr2function        | 'assemble'            | How to convert Expression to Function. Allowed values:            |
        |                      |                       |                                                                   |
        |                      |                       | - 'assemble'                                                      |
        |                      |                       | - 'project'                                                       |
        |                      |                       | - 'interpolate'                                                   |
        +----------------------+-----------------------+-------------------------------------------------------------------+
        | finalize             | False                 | Switch whether to finalize if Field. This is especially useful    |
        |                      |                       | when a costly computation is only interesting at the end time.    |
        +----------------------+-----------------------+-------------------------------------------------------------------+

        """
        params = ParamDict(
            # Configure direct compute requests through timestep counting
            start_timestep = -1e16,
            end_timestep = 1e16,
            stride_timestep = 1,

            # Configure direct compute requests through physical time intervals
            start_time = -1e16,
            end_time = 1e16,
            stride_time = 1e-16,

            # Trigger and configure action after each direct compute request
            plot = False,
            plot_args={},
            safe = True,
            save = False,
            save_as = cls.default_save_as(),

            #callback = False,

            # Configure computing
            expr2function = "assemble",
            #project = False, # This is the safest approach
            #assemble = True, # This is faster but only works for for DG0
            #interpolate = False, # This will be the best when properly implemented in fenics

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
    
    @name.setter
    def name(self, value):
        """Set name property"""
        self._name = value

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

    def explicit_dependencies(self):
        """Can be overloaded to explicitly specify dependencies. This is useful for complex
        fields where the normal dependency-checks fail.
        """
        return None

    # --- Helper functions

    def expr2function(self, expr, function):
        """ Convert an expression into a function. How this is done is
        determined by the parameters (assemble, project or interpolate).
        """
        space = function.function_space()

        if self.params.expr2function == "assemble":
            # Compute average values of expr for each cell and place in a DG0 space
            test = TestFunction(space)
            scale = 1.0 / CellVolume(space.mesh())
            assemble(scale*inner(expr, test)*dx, tensor=function.vector())
            return function

        elif self.params.expr2function == "project":
            # TODO: Avoid superfluous function creation with fenics-dev/1.5 by using:
            #project(expr, space, function=function)
            function.assign(project(expr, space))
            return function

        elif self.params.expr2function == "interpolate":
            # TODO: Need interpolation with code generated from expr, waiting for uflacs work.
            function.interpolate(expr) # Currently only works if expr is a single Function
            return function

        else:
            error("No action selected, need to choose either assemble, project or interpolate.")


    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        result.params = ParamDict(self.params)
        return result

    def __add__(self, o):
        from cbcpost.metafields import Add
        return Add(self,o)

    def __radd__(self, o):
        from cbcpost.metafields import Add
        return Add(o, self)

    def __mul__(self, o):
        from cbcpost.metafields import Multiply
        return Multiply(self, o)

    def __rmul__(self, o):
        from cbcpost.metafields import Multiply
        return Multiply(o, self)

    def __div__(self, o):
        from cbcpost.metafields import Divide
        return Divide(self,o)

    def __rdiv__(self, o):
        from cbcpost.metafields import Divide
        return Divide(o, self)

    def __sub__(self,o):
        from cbcpost.metafields import Subtract
        return Subtract(self,o)

    def __rsub__(self,o):
        from cbcpost.metafields import Subtract
        return Subtract(o, self)

    @classmethod
    def start_recording(cls):
        assert cls._recording == False, "Recording already started!"
        cls._recording = True

    @classmethod
    def stop_recording(cls):
        cls._recording = False
        records = cls._record
        cls._record = []
        return records
