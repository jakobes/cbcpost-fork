Utilities
=============================================
Functions
_____________________________________________

boundarymesh_to_mesh_dofmap
---------------------------------------------
Find the mapping from dofs on boundary FS to dofs on full mesh FS


cbc_log
---------------------------------------------
Log on master process.


cbc_print
---------------------------------------------
Print on master process.


cbc_warning
---------------------------------------------
Raise warning on master process.


compute_connectivity
---------------------------------------------
Compute connected regions of mesh.
Regions are considered connected if they share a vertex through an edge.


create_function_from_metadata
---------------------------------------------
Create a function from metadata


create_slice
---------------------------------------------
Create a slicemesh from a basemesh.



Arguments:

:basemesh: Mesh to slice
:point: Point in slicing plane
:normal: Normal to slicing plane
:closest_region: Set to True to extract disjoint region closest to specified point
:crinkle_clip: Set to True to return mesh of same topological dimension as basemesh

.. note::

Only 3D-meshes currently supported for slicing.

.. warning::

Slice-instances are intended for visualization only, and may produce erronous
results if used for computations.


create_submesh
---------------------------------------------
This function allows for a SubMesh-equivalent to be created in parallel


get_memory_usage
---------------------------------------------
Return memory usage in MB


get_set_vector
---------------------------------------------
Equivalent of setvector[set_indices] = getvector[get_indices] for global indices (MPI-blocking).
Pass temp_array to avoid initiation of array on call.


import_fenicstools
---------------------------------------------
Import fenicstools helper function.


in_serial
---------------------------------------------
Return True if running in serial.


mesh_to_boundarymesh_dofmap
---------------------------------------------
Find the mapping from dofs on full mesh FS to dofs on boundarymesh FS


on_master_process
---------------------------------------------
Return True if on process number 0.


restriction_map
---------------------------------------------
Return a map between dofs in Vb to dofs in V. Vb's mesh should be a submesh of V's Mesh.


safe_mkdir
---------------------------------------------
Create directory without exceptions in parallel.


strip_code
---------------------------------------------
Strips code of unnecessary spaces, comments etc.


time_to_string
---------------------------------------------
Format time in seconds as a human readable string.


timeit
---------------------------------------------
Simple timer

Classes
_____________________________________________

Loadable
---------------------------------------------
Create an instance that reads a Field from file as specified by the
parameters. Requires that the file is written in cbcpost (or in the same format).



Arguments:

:filename: Filename where function is stored
:fieldname: Name of Field
:timestep: Timestep to load
:time: Time
:saveformat: Saveformat of field
:s function: Function to load Field into

This class is used internally from :class:'.Replay' and :class:'Restart',
and made to be passed to *PostProcessor.update_all*.


Slice
---------------------------------------------
Deprecated Slice-class


Timer
---------------------------------------------
Class to perform timing.



Arguments:

:frequency: Frequency which to report timings.

