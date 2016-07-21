Postprocessor
=============================================

PostProcessor
---------------------------------------------
All basic user interface is gathered here.



Default parameters are:


.. tabularcolumns:: p{5cm}p{4cm}p{6cm}

+----------------------+-----------------------+--------------------------------------------------------------+
|Key                   | Default value         |  Description                                                 |
+======================+=======================+==============================================================+
| casedir              | '.'                   | Case directory - relative path to use for saving             |
+----------------------+-----------------------+--------------------------------------------------------------+
| extrapolate          | True                  | Constant extrapolation of fields prior to first              |
|                      |                       | update call                                                  |
+----------------------+-----------------------+--------------------------------------------------------------+
| initial_dt           | 1e-5                  | Initial timestep. Only used in planning algorithm at first   |
|                      |                       | update call.                                                 |
+----------------------+-----------------------+--------------------------------------------------------------+
| clean_casedir        | False                 | Clean out case directory prior to update.                    |
+----------------------+-----------------------+--------------------------------------------------------------+
| flush_frequency      | 1                     | Frequency to flush shelve and txt files (playlog,            |
|                      |                       | metadata and data)                                           |
+----------------------+-----------------------+--------------------------------------------------------------+


Planner
---------------------------------------------
Planner class to plan for all computations.


Saver
---------------------------------------------
Class to handle all saving in cbcpost.


Plotter
---------------------------------------------
Class to handle plotting of objects.

Plotting is done using pylab or dolfin, depending on object type.

