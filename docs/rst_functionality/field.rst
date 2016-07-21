The :class:`Field`-class and subclasses
---------------------------------------------
To understand how cbcpost works, one first needs to understand the role of *Fields*. All desired postprocessing must be added to the PostProcessor as subclasses of :class:`.Field`. The class itself is to be considered as an abstract base class, and must be subclassed to make sense.

All subclasses are expected to implement (at minimum) the :meth:`.Field.compute`-method. This takes a single argument which can be used to retrieve dependencies from other fields.

An important property of the :class:`.Field`-class, is the parameters. Through the :class:`.Parameterized`-interface, it implements a set of default parameters that is used by the PostProcessor when determining how to handle any given Field, with respect to computation frequency, saving and plotting.


Subclassing the :class:`Field`-class
````````````````````````````````````````````````````

To compute any quantity of interest, one needs to either use one of the provided metafields or subclass :class:`Field`. In the following, we will first demonstrate the simplicity of the interface, before demonstrating the flexibility of it.

.. _stress-tensor:

A viscous stress tensor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The viscous stress tensor for a Newtonian fluid is computed as

.. math::

    \sigma(\mathbf{u}, p) = -p\mathbb{I}+\mu(\nabla \mathbf{u}+\nabla \mathbf{u}^T)

where :math:`\mu` is the dynamic viscosity, :math:`\mathbf{u}` is the fluid velocity and :math:`p` is the pressure. A Field to compute this might be specified as the following:

.. highlight:: python
.. code-block:: python

    from dolfin import *
    from cbcpost import Field
    from cbcpost.spacepool import get_grad_space
    class Stress(Field):
        def __init__(self, mu, params=None, name="default", label=None):
            Field.__init__(self, params, name, label)
            self.mu = mu

        def before_first_compute(self, get):
            u = get("Velocity")

            # Create Function container on space of velocity gradient
            V = get_grad_space(u)
            self._function = Function(V, name=self.name)

        def compute(self, get):
            u = get("Velocity")
            p = get("Pressure")
            mu = self.mu

            expr = - p*Identity(u.cell().d) + mu*(grad(u)+grad(u)^T)

            return self.expr2function(expr, self._function)


Note that we have overridden three methods defined in :class:`.Field`:

- __init__
- before_first_compute
- compute

The __init__ method is only used to pass any additional arguments to our Field, in this case the viscosity. The keyword arguments *params*, *name* and *label* are passed directly to :meth:`.Field.__init__`.

before_first_compute is used to do any costly computations or allocations that are only required once. This is called from the postprocessor before any calls to compute is made. In this case we create a container (*_function*) that we can later use to store our computations. We use the *get*-argument to fetch the field named *Velocity*, and the helper function :func:`.get_grad_space` to get the gradient space of the Velocity (a TensorFunctionSpace).

The compute method is responsible for computing our quantity. This is called from the postprocessor every time the :class:`.Planner` determines that this field needs to be computed. Here we use the *get*-argument to fetch the *Velocity* and *Pressure* required to compute the stress. We formulate the stress, and converts to a function using the helper function :meth:`.Field.expr2function`.


Computing the maximum pressure drop
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In this next section, we demonstrate some more functionality one can take advantage of when subclassing the :class:`Field`-class. In a flow, the maximum pressure drop gives an indication of the forces involved in the flow. It can be written as

.. math::

    \tilde{p} := \max_{t \in [ 0,T ]} ( \max_{\mathbf{x} \in \Omega} p(\mathbf{x}, t) - \min_{\mathbf{x} \in \Omega} p(\mathbf{x}, t) )


A :class:`.Field`-class to compute this can be implemented as

.. highlight:: python
.. code-block:: python

    from dolfin import *
    from cbcpost import Field
    from cbcpost.spacepool import get_grad_space
    class PTilde(Field):
        def add_fields(self):
            return [ Maximum("Pressure"), Minimum("Pressure") ]

        def before_first_compute(self, get):
            self._ptilde = 0.0
            self._tmax = 0.0

        def compute(self, get):
            pmax = get("Maximum_Pressure")
            pmin = get("Minimum_Pressure")
            t = get("t")

            if pmax-pmin > self._ptilde:
                self._ptilde = pmax-pmin
                self._tmax = t

            return None

        def after_last_compute(self, get):
            return (self._ptilde, self._tmax)

Here, we implement two more :class:`.Field`-methods:

- add_fields
- after_last_compute

The add_fields method is a convenience function to make sure that dependent Fields are added to the postprocessor. This can also be handled manually, but this makes for a cleaner code. Here we add two fields to compute the (spatial) :class:`.Maximum` and :class:`.Minimum` of the pressure.

The method after_last_compute is called when the compution is finished. This is determined by the time parameters (see :ref:`field-parameters`), and handled within the postprocessors :class:`.Planner`-instance.


Field names
````````````````````
The internal communication of fields is based on the name of the :class:`.Field`-instances. The default name is ::

    [class name]-[optional label]

The label can be specified in the *__init__*-method (through the *label*-keyword), or a specific name can be set using the *name*-keyword.

When subclassing the :class:`.Field`-class, the default naming convention can overloaded in the :attr:`.Field.name`-property.

The *get*-argument
````````````````````````````````````
In the three methods *before_first_compute*, *compute* and *after_last_compute* a single argument (in addition to *self*) is passed from the postprocessor, namely the *get*-argument. This argument is used to fetch the computed value from other fields, through the postprocessor. The argument itself points to the :meth:`.PostProcessor.get`-method, and is typically used with these two arguments:

- Field name
- Relative timestep

A call using the *get*-function will trigger a computation of the field with the given name, and cache it in the postprocessor. Therefore, a second call with the same arguments, will return the cached value and not trigger a new computation.

The calls to the *get*-function also determines the dependencies of a Field (see :ref:`dependency-handling`).




.. _field-parameters:

Parameters
````````````````````````````````````
The logic of the postprocessor relies on a set of parameters defined on each Field. For explanation of the common parameters and their default, see :meth:`.Field.default_params()`.


SolutionField
``````````````
The :class:`.SolutionField`-class is a convenience class, for specifying Field(s) that will be provded as solution variables. It requires a single argument as the name of the Field. Since it is a solution field, it does not implement it does not implement a *compute*-method, but relies on data passed to the :meth:`.PostProcessor.update_all` for its associatied data. It is used to be able to build dependencies in the postprocessor.


MetaField and MetaField2
``````````````````````````
Two additional base classes are also available. These are designed to allow for computations that are not specific (such as PTilde or Stress), but where you need to specify the Field(s) to compute on.

Subclasses of the :class:`.MetaField`-class include for example :class:`.Maximum`, :class:`.Norm` and :class:`.TimeIntegral`, and takes a single name (or Field) argument to specify which Field to do the computation on.

Subclasses of the :class:`.MetaField2` include :class:`.ErrorNorm`, and takes two name (or Field) arguments to specify which Fields to compute with.

Provided fields
`````````````````
Several meta fields are provided in cbcpost, for general computations. These are summarized in the following table:

=====================     ==========================      =======================    =================
**Time dependent**        **Spatially restricted**        **Norms and averages**     **Other**
---------------------     --------------------------      -----------------------    -----------------
TimeDerivative            SubFunction                     DomainAvg                  Magnitude
TimeIntegral              Restrict                        Norm
TimeAverage               Boundary                        ErrorNorm
                          PointEval                       Maximum
                                                          Minimum
=====================     ==========================      =======================    =================


For more details of each field, refer to :ref:`metafields`.

