Installation
============

Dependencies
____________

The installation of cbcpost requires the following environment:

    * Python 2.7
    * Numpy
    * Scipy
    * Any dbm compatible database (dbhash, dbm or gdbm)
    * `FEniCS <http://fenicsproject.org>`_
    * fenicstools (optional but highly recommended, tools to inspect parts of a solution)

To install FEniCS, please refer to the `FEniCS download page
<http://fenicsproject.org/download/>`_.

To install fenicstools, please refer to the `github page
<http://github.org/mikaem/fenicstools>`_.

cbcpost and fenicstools follows the same version numbering as FEniCS,
so make sure you install the matching versions.
Backwards compatibility is not guaranteed (and quite unlikely).

In addition, to run the test suite

    * pytest >2.4.0


Installing
__________

Get the software with git and install using pip:

.. code-block:: bash

   git clone https://bitbucket.org/simula_cbc/cbcpost.git
   cd cbcpost
   pip install .

See the pip documentation for more installation options.
