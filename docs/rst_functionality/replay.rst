Replay
------------------------
One of the key functionalities of the cbcpost framework is the ability to replay problem. Consider the case where one wants to extract additional information from a simulation. Simulations are typically costly, and redoing simulations are not generally desired (or even feasible). This motivates the functionality to *replay* the simulation by loading the computed solution back into memory and compute additional fields.

This has several major benefits:

- Compute additional quantities
- Limit memory consumption of initial computation
- Compute quantities unsupported in parallel
- Compute costly, conditional quantities (e.g. not to be performed if simulation was unable to complete)
- Create visualization data

The interface to the replay module is minimal:

.. code-block:: python
    
    from cbcpost import PostProcessor, Replay

    pp = PostProcessor(dict(casedir="ExistingResults/"))
    pp.add_field(MyCustomField(), dict(save=True))
    
    replayer = Replay(pp)
    replayer.replay()
    

In the replay module, all fields that are stored in a reloadable format will be treated as a solution. They will be passed to a postprocessor as instances of the :class:`.Loadable`-class. This makes sure that no unnecessary I/O-operations occur, as the stored data are only loaded when they are triggered in the postprocessor.




