Functionality
===========================

The main functionality is handled with a :class:`.PostProcessor`-instance, populated with several :class:`.Field`-items.

The :class:`.Field`-items added to the :class:`PostProcessor` can represent *meta* computations (:class:`.MetaField`, :class:`.MetaField2`) such as time integrals or time derivatives, restrictions or subfunction, or norms. They can also represent custom computations, such as stress, strain, stream functions etc. All subclasses of the :class:`.Field`-class inherits a set of parameters used to specify computation logic, and has a set of parameters related to saving, plotting, and computation intervals.

The :class:`.Planner`, instantiated by the PostProcessor, handles planning of computations based on Field-parameters. It also handles the dependency, and plans ahead for computations at a later time.

For saving purposes the PostProcessor also creates a :class:`Saver`-instance. This will save Fields as specified by the Field-parameters and computed fields. It saves in a structured manner within a specified case directory.

In addition, there is support for plotting in the :class:`.Plotter`-class, also created within the PostProcessor. It uses either dolfin.plot or pyplot.plot to plot data, based on data format.

.. toctree::
   :maxdepth: 2

   field
   postprocessor
   replay
   restart
   utilities





