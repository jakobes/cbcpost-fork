Restart
=============================================

Restart
---------------------------------------------
Class to fetch restart conditions through.


Default parameters are:


.. tabularcolumns:: p{5cm}p{4cm}p{6cm}

+----------------------+-----------------------+-------------------------------------------------------------------+
|Key                   | Default value         |  Description                                                      |
+======================+=======================+===================================================================+
| casedir              | '.'                   | Case directory - relative path to read solutions from             |
+----------------------+-----------------------+-------------------------------------------------------------------+
| restart_times        | -1                    | float or list of floats to find restart times from. If -1,        |
|                      |                       | restart from last available time.                                 |
+----------------------+-----------------------+-------------------------------------------------------------------+
| solution_names       | 'default'             | Solution names to look for. If 'default', will fetch all          |
|                      |                       | fields stored as SolutionField.                                   |
+----------------------+-----------------------+-------------------------------------------------------------------+
| rollback_casedir     | False                 | Rollback case directory by removing all items stored after        |
|                      |                       | largest restart time. This allows for saving data from a          |
|                      |                       | restarted simulation in the same case directory.                  |
+----------------------+-----------------------+-------------------------------------------------------------------+

