.. cbcpost documentation master file, created by
   sphinx-quickstart on Tue Feb  4 08:51:54 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

cbcpost - a postprocessing framework for FEniCS
=================================================

cbcpost is developed to simplify the postprocessing of simulation results, produced by FEniCS solvers.

The framework is designed to take any given solution, and compute and save any derived data. Derived data can easily be made highly complex, due to the modular design and implementation of computations of quantities such as integrals, derivatives, magnitude etc, and the ability to chain these.

The interface is designed to be simple, with minimal cluttering of a typical solver code. This is illustrated by the following simple example:

.. code-block :: python

    # ... problem set up ...

    # Set up postprocessor
    solution = SolutionField("Displacement", dict(save=True))
    postprocessor = PostProcessor(dict(casedir="Results/"))
    postprocessor.add_field(solution)
    
    # Add derived fields
    postprocessor.add_fields([
        Maximum("Displacement", dict(save=True)),
        TimeAverage("Displacement", dict(save=True, start_time=1.0,
                                        end_time=2.0)),
    ])

    t = 0.0
    timestep = 0
    while t < T:
        timestep += 1
        # ... solve equation ...

        # Update postprocessor
        postprocessor.update_all(dict("Displacement"=lambda: u), timestep, t)

        # continue

cbcpost is developed at the `Center for Biomedical Computing <http://cbc.simula.no/pub/>`_, at `Simula Research Laboratory <https://www.simula.no/>`_ by `Øyvind Evju <https://www.simula.no/people/oyvinev>`_ and `Martin Sandve Alnæs <https://www.simula.no/people/martinal>`_.

**Contents:**

.. toctree::
   :maxdepth: 2
   :numbered:

   installation
   features
   Demos/index
   rst_functionality/index
   rst_programmers_reference/index
   contributing
