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
Handle plotting of all fields where this is requested.

Uses dolfin.plot to plot dolfin-objects (typically Functions), and pylab.plot
to plot single scalars (float, int).

This code is intended for internal usage, and is called from a PostProcessor
instance.
"""

from cbcpost.utils import in_serial, cbc_warning
import os

from dolfin import Function, plot


def disable_plotting():
    "Disable all plotting if we run in parallell."
    if disable_plotting.value == "init":
        if in_serial() and 'DISPLAY' in os.environ:
            disable_plotting.value = False
        elif 'DISPLAY' not in os.environ:
            cbc_warning("Did not find display. Disabling plotting.")
            disable_plotting.value = True
        else:
            cbc_warning("Unable to plot in paralell. Disabling plotting.")
            disable_plotting.value = True

    return disable_plotting.value
disable_plotting.value = "init"

def import_pylab():
    "Set up pylab if available."
    if import_pylab.value == "init":
        if disable_plotting():
            import_pylab.value = None
        else:
            try:
                import pylab
                pylab.ion()
                import_pylab.value = pylab
            except ImportError:
                cbc_warning("Unable to load pylab. Disabling pylab plotting.")
                import_pylab.value = None
    return import_pylab.value
import_pylab.value = "init"


class Plotter():
    """Class to handle plotting of objects.

    Plotting is done using pylab or dolfin, depending on object type."""
    def __init__(self, timer):
        self._timer = timer

        # Cache for plotting
        self._plot_cache = {}

    def _plot_dolfin(self, t, timestep, field, data):
        "Plot field using dolfin plot command"
        # Plot or re-plot
        plot_object = self._plot_cache.get(field.name)
        if plot_object is None:
            plot_object = plot(data, title=field.name, **field.params.plot_args)
            self._plot_cache[field.name] = plot_object
        else:
            plot_object.plot(data)

        # Set title and show
        title = "%s, t=%0.4g, timestep=%d" % (field.name, t, timestep)
        plot_object.parameters["title"] = title

    def _plot_pylab(self, t, timestep, field, data):
        "Plot using pylab if field is a single scalar."
        pylab = import_pylab()
        if not pylab:
            return

        # Get current time
        #t = self.get("t")
        #timestep = self.get('timestep')

        # Values to plot
        x = t
        y = data

        # Plot or re-plot
        plot_data = self._plot_cache.get(field.name)
        if plot_data is None:
            figure_number = len(self._plot_cache)
            pylab.figure(figure_number)

            xdata = [x]
            ydata = [y]
            newmin = min(ydata)
            newmax = max(ydata)

            plot_object, = pylab.plot(xdata, ydata)
            self._plot_cache[field.name] = (plot_object, figure_number,
                                            newmin, newmax)
        else:
            plot_object, figure_number, oldmin, oldmax = plot_data
            pylab.figure(figure_number)

            xdata = list(plot_object.get_xdata())
            ydata = list(plot_object.get_ydata())
            xdata.append(x)
            ydata.append(y)
            newmin = 1.2*min(ydata)
            newmax = 1.2*max(ydata)

            # Heuristics to avoid changing axis bit by bit, which results in
            # fluttering plots. (Based on gut feeling, feel free to adjust
            # these if you have a use case it doesnt work for)
            if newmin < oldmin:
                # If it has decreased, decrease by at least this factor
                #ymin = min(newmin, oldmin*0.8) # TODO: Negative numbers?
                ymin = newmin
            else:
                ymin = newmin
            if newmax > oldmax:
                # If it has increased, increase by at least this factor
                #ymax = max(newmax, oldmax*1.2) # TODO: Negative numbers?
                ymax = newmax
            else:
                ymax = newmax

            # Need to store min/max for the heuristics to work
            self._plot_cache[field.name] = (plot_object, figure_number,
                                            ymin, ymax)

            plot_object.set_xdata(xdata)
            plot_object.set_ydata(ydata)

            pylab.axis([xdata[0], xdata[-1], ymin, ymax])

        # Set title and show
        title = "%s, t=%0.4g, timestep=%d, min=%.2g, max=%.2g" % (field.name,
                                                                  t, timestep,
                                                                  newmin,
                                                                  newmax)
        plot_object.get_axes().set_title(title)
        pylab.xlabel("t")
        pylab.ylabel(field.name)
        pylab.draw()


    def _action_plot(self, t, timestep, field, data):
        "Apply the 'plot' action to computed field data."
        if data == None:
            return

        if disable_plotting():
            return
        if isinstance(data, Function):
            self._plot_dolfin(t, timestep, field, data)
        elif isinstance(data, float):
            self._plot_pylab(t, timestep, field, data)
        else:
            cbc_warning("Unable to plot object %s of type %s."
                        % (field.name, type(data)))
        self._timer.completed("PP: plot %s" %field.name)



    def update(self, t, timestep, cache, triggered_or_finalized):
        """Update plot windows with fields that have been triggered or finalized
        at the current timestep."""
        for field in triggered_or_finalized:
            #print name, cache[name]
            if field.params.plot:
                self._action_plot(t, timestep, field, cache[field.name])
