.. _Replay:

Replay a Problem
========================================

Once a simulation is completed, one might want to compute other fields of the solution. This can be
done with cbcposts :class:`.Replay`-functionality. The process can be done in very few lines of code.

In the following, we initialize a replay of the heat equation solved in :ref:`Basic` and restarted in :ref:`Restart`. First, we set up a postprocessor with the fields we wish to compute: ::

	from cbcpost import *
	from dolfin import set_log_level, WARNING, interactive
	set_log_level(WARNING)

	pp = PostProcessor(dict(casedir="../Basic/Results"))

	pp.add_fields([
	    SolutionField("Temperature", dict(plot=True)),
	    Norm("Temperature", dict(save=True, plot=True)),
	    TimeIntegral("Norm_Temperature", dict(save=True, start_time=0.0,
											  end_time=6.0)),
	])

To *replay* the simulation, we do: ::

	replayer = Replay(pp)
	replayer.replay()
	interactive()

.. note: This functionality is currently only supported in serial.
