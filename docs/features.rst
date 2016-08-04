Features
========

The core concept in cbcpost is the :class:`.Field`, which represents
something that can be computed from simulation solutions or other fields.
The main features of cbcpost are

- Saving in 7 different save formats (xdmf, hdf5, xml, xml.gz, pvd, shelve, txt)
- Plotting using dolfin.plot or pyplot
- Automatic planning of field computations, saving and plotting
- Automatic dependency handling
- A range of predefined fields built in, including time integrals, point evaluations and norms
- Easily expandable with custom :class:`.Field`-subclasses
- Compute fields during simulation or replay from results on file
- Restart support
- Flexible parameter system
- Small footprint on solver code
