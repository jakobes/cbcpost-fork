

Batch running (cbcbatch)
--------------------------------
When you've set up a simulation in a python file, you can investigate a range of parameters through the shell script *cbcbatch*.
This allows you to easily run simulations required to for example compute convergence rates or parameter sensitivity with respect to some compute Field.

Based on the parameters of your solver or problem, you can set up parameter ranges with command line arguments to cbcbatch. Say for example that you wish to
investigate the effects of refinement level *N* and timestep *dt* over a given range. Then you can launch cbcbatch by invoking

.. code-block:: bash

    cbcbatch run.py N=[8,16,32,64,128] dt=[0.1,0.05,0.025,0.0125] \
        casedir=BatchResults

where run.py is the python file to launch the simulation. This will then add all combinations of N and dt (5*4=20) to a queue, and launch simulations when the
resources are available. We call *dt* and *N* the *batch parameters*.

By default, cbcbatch runs on a single core, but this can be modified by setting the *num_cores* argument:

.. code-block:: bash

    cbcbatch run.py N=[8,16,32,64,128] dt=[0.1,0.05,0.025,0.0125] \
        casedir=BatchResults num_cores=8

This will cause 8 simulations to be run at a time, and new ones started as soon as one core becomes available. Since there may be a large variations in computational
cost between parameters, it is also supported to tie one of the batch parameters to run in parallel with mpirun:

.. code-block:: bash

    cbcbatch run.py N=[8,16,32,64,128] dt=[0.1,0.05,0.025,0.0125] \
        casedir=BatchResults num_cores=8 mpirun=[1,1,2,4,8] \
        mpirun_parameter=N

This command will run all simulations with N=1 and N=2 on a single core, N=32 on 2 cores, N=64 on 4 cores and N=128 on 8 cores.

.. important:: The runnable python file must set *set_parse_command_line_arguments(True)* to be run in batch mode.

.. important:: The command line parameters *casedir*, *mpirun*, *mpirun_parameter* and *num_cores* are reserved for cbcbatch and can thus not be used as batch parameters.