.. _Restart:

Restart a Problem
========================================

Say we wish to run our simulation further than t=3.0, to see how it develops. To restart a problem, all you
need is to use the computed solution as initial conditions in a simiular problem setup.

Restarting the heat equation solved as in :ref:`Basic`, can be done really simple
with cbcpost. Starting with the python-file in :ref:`Basic`, we only have to make a couple of minor
changes.

We change the parameters T0 and T to look at the interval :math:`t \in [3,6]`: ::

    params.T0 = 3.0
    params.T = 6.0

and we replace the initial condition, using the :class:`.Restart`-class: ::
    
    # Get restart data
    restart = Restart(dict(casedir="../Basic/Results/"))
    restart_data = restart.get_restart_conditions()
    
    # Initial condition
    U = restart_data.values()[0]["Temperature"]

Note that we point :class:`.Restart` to the case directory where the solution is stored. We could also choose
to write our restart data to the same directory when setting up the postprocessor: ::

    pp = PostProcessor(dict(casedir="../Basic/Results"))
