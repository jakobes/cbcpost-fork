
Restart
---------------------

The restart functionality lets the user set up a problem for restart. This functionality is based on the idea that a restart of a simulation is nothing more than changing the initial conditions of the problem in question. Therefore, the :class:`.Restart`-class is used to extract the solution at any given time(s) in a format that may be used as intiial conditions.

If we want to restart any problem, where a solution has been stored by cbcpost, we can simply point to the
case directory:

.. highlight:: python
.. code-block:: python

    from cbcpost import *
    restart = Restart(dict(casedir='Results/'))
    restart_data = restart.get_restart_conditions()

If you for instance try to restart the simple case of the heat equation, *restart_data* will be a *dict* of
the format {t0: {"Temperature": U0}}. If you try to restart for example a (Navier-)Stokes-problem, it will take
a format of {t0: {"Velocity": U0, "Pressure": P0}}.

There are several options for fetching the restart conditions.

Specify restart time
`````````````````````````````````````````

You can easily specify the restart time to fetch the solution from:

.. highlight:: python
.. code-block:: python

    t0 = 2.5
    restart = Restart(dict(casedir='Results/', restart_times=t0))
    restart_data = restart.get_restart_conditions()

If the restart time does not match a solution time, it will do a linear interpolation between the closest
existing solution times.

Fetch multiple restart times
`````````````````````````````````````````

For many problems, initial conditions are required at several time points
prior to the desired restart time. This can be handled through:

.. highlight:: python
.. code-block:: python

    dt = 0.01
    t1 = 2.5
    t0 = t1-dt
    restart = Restart(dict(casedir='Results/', restart_times=[t0,t1]))
    restart_data = restart.get_restart_conditions()


Rollback case directory for restart
`````````````````````````````````````````

If you wish to write the restarted solution to the same case directory, you will need to clean up the case
directory to avoid write errors. This is done by setting the parameter *rollback_casedir*:

.. highlight:: python
.. code-block:: python

    t0 = 2.5
    restart = Restart(dict(casedir='Results/', restart_times=t0,
                           rollback_casedir=True))
    restart_data = restart.get_restart_conditions()

Specifying solution names to fetch
`````````````````````````````````````````

By default, the Restart-module will search through the case directory for all data stored as a
:class:`SolutionField`. However, you can also specify other fields to fetch as restart data:

.. highlight:: python
.. code-block:: python

    solution_names = ["MyField", "MyField2"]
    restart = Restart(dict(casedir='Results/', solution_names=solution_names))
    restart_data = restart.get_restart_conditions()

In this case, all :class:`SolutionField`-names will be ignored, and only restart conditions from fields
named *MyField* and *MyField2* will be returned.


Changing function spaces
`````````````````````````````````````````
If you wish to restart the simulation using different function spaces, you can pass the function spaces
to *get_restart_conditions*:

.. highlight:: python
.. code-block:: python

    V = FunctionSpace(mesh, "CG", 3)
    restart = Restart(dict(casedir='Results/'))
    restart_data = restart.get_restart_conditions(spaces={"Temperature": V})

.. note:: This does not currently work for function spaces defined on a different mesh.
