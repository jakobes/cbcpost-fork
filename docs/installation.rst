Installation
=====================================

Quick Install
_________________________________

Install using git clone:

.. code-block:: bash

   git clone https://bitbucket.org/simula_cbc/cbcpost.git
   cd cbcpost
   python setup.py install

Install using pip:

.. code-block:: bash

   pip install git+https://bitbucket.org/simula_cbc/cbcpost.git


Dependencies
__________________________________

The installation of cbcpost requires the following environment:
    * Python 2.7
    * Numpy
    * Scipy
    * (`FEniCS <http://fenicsproject.org>`_) 1.4.0

To install FEniCS, please refer to the `FEniCS download page
<http://fenicsproject.org/download/>`_. cbcpost follows the same version numbering
as FEniCS, so make sure you install the correct FEniCS version. Backwards
compatibility is not guaranteed (and quite unlikely).

In addition, cbcpost can utlize other libraries for added functionality
   * fenicstools 1.4.0 (highly recommended, tools to inspect parts of a solution)
   * mpi4py
   * pytest >2.4.0 (required to run test suite)

fenicstools can be installed using pip:

.. code-block:: bash

   pip install https://github.com/mikaem/fenicstools/archive/v1.4.0.zip

