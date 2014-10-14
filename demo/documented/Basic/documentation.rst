.. _Basic:

A Basic Use Case
========================================

To demonstrate the functionality of the postprocessor, consider the 3D-case of the heat equation with
variable diffusivity. The full demo can be found in :download:`Basic.py`.

The general heat equation reads

..  math::
    \frac{\partial u}{\partial t} + \alpha(x) \Delta u = f

where u typically denotes the temperature and :math:`\alpha` denotes the material diffusivity.

Boundary conditions are in our example given as

.. math::
    u(x,t) = Asin(2\pi tx_0), x \in \partial \Omega
    
and initial condition

.. math::
    u(x,0) = 0.
    
We also use f=0, and solve the equations at the unit cube for :math:`t \in (0,3]`.

Setting up the problem
_______________________________________________

We start by defininge a set of parameters for our problem: ::

    from cbcpost import *
    from cbcpost.utils import cbc_print
    from dolfin import *
    set_log_level(WARNING)
    
    # Create parameters for problem
    params = ParamDict(
        T = 3.0,            # End time
        dt = 0.05,          # Time step
        theta = 0.5,        # Time stepping scheme (0.5=Crank-Nicolson)
        alpha0 = 10.0,      # Outer diffusivity
        alpha1 = 1e-3,      # Inner diffusivity
        amplitude = 3.0,    # Amplitude of boundary condition
    )


The parameters are created using the utility class :class:`.ParamDict`, which extend the built-in python
dict.

We the use the parameters to set up the problem using FEniCS: ::

    # Create mesh
    mesh = UnitCubeMesh(21,21,21)
    
    # Function spaces
    V = FunctionSpace(mesh, "CG", 1)
    u,v = TrialFunction(V), TestFunction(V)
    
    # Time and time-stepping
    t = 0.0
    timestep = 0
    dt = Constant(params.dt)
    
    # Initial condition
    U = Function(V)
    
    # Define inner domain
    def inside(x):
        return (0.5 < x[0] < 0.8) and (0.3 < x[1] < 0.6) and (0.2 < x[2] < 0.7)
    
    class Alpha(Expression):
        "Variable conductivity expression"
        def __init__(self, alpha0, alpha1):
            self.alpha0 = alpha0
            self.alpha1 = alpha1
        
        def eval(self, value, x):
            if inside(x):
                value[0] = self.alpha1
            else:
                value[0] = self.alpha0
    
    # Conductivity
    alpha = project(Alpha(params.alpha0, params.alpha1), V)
    
    # Boundary condition
    u0 = Expression("ampl*sin(x[0]*2*pi*t)", t=t, ampl=params.amplitude)
    bc = DirichletBC(V, u0, "on_boundary")
    
    # Source term
    f = Constant(0)
    
    # Bilinear form
    a = 1.0/dt*inner(u,v)*dx() + Constant(params.theta)*alpha*inner(grad(u), grad(v))*dx()
    L = 1.0/dt*inner(U,v)*dx() + Constant(1-params.theta)*alpha*inner(grad(U), grad(v))*dx() + inner(f,v)*dx()
    A = assemble(a)
    b = assemble(L)
    bc.apply(A)


Setting up the PostProcessor
__________________________________
To set up the use case, we specify the case directory, and asks to clean out the case directory if there
is any data remaining from a previous simulation: ::
    
    pp = PostProcessor(dict(casedir="Results", clean_casedir=True))
    
Since we`re solving for temperature, we add a SolutionField to the postprocessor: ::

    pp.add_field(SolutionField("Temperature", dict(save=True,
                                    save_as=["hdf5", "xdmf"],
                                    plot=True,
                                    plot_args=dict(range_min=-params.amplitude, range_max=params.amplitude),
                                    )))
                                                   
Note that we pass parameters, specifying that the field is to be saved in hdf5 and xdmf formats. These
formats are default for dolfin.Function-type objects. We also ask for the Field to be plotted, with plot_args
specifying the plot window. These arguments are passed directly to the dolfin.plot-command.

Time derivatives and time integrals
-----------------------------------------
We can compute both integrals and derivatives of other Fields. Here, we add the integral of temperature from
t=1.0 to t=2.0, the time-average from t=0.0 to t=5.0 as well as the derivative of the temperature field. ::

    pp.add_fields([
        TimeIntegral("Temperature", dict(save=True, start_time=1.0, end_time=2.0)),
        TimeAverage("Temperature", dict(save=True, end_time=params.T)),
        TimeDerivative("Temperature", dict(save=True)),
        ])

Again, we ask the fields to be saved. The save formats are decided by the datatype returned from the
*compute*-functions.

Inspecting parts of a solution
-----------------------------------------------
We can also define fields to inspect parts of other fields. For this, we use some utilities from
:class:`.cbcpost.utils`.
For this problem, the domain of a different diffusivity lies entirely within the unit cube, and thus it may
make sense to view some of the interior. We start by creating (sub)meshes of the domains we wish to inspect: ::

    from cbcpost.utils import create_submesh, Slice
    celldomains = CellFunction("size_t", mesh)
    celldomains.set_all(0)
    AutoSubDomain(inside).mark(celldomains, 1)
    
    slicemesh = Slice(mesh, (0.7,0.5,0.5), (0.0,0.0,1.0))
    submesh = create_submesh(mesh, celldomains, 1)

We then add instances of the fields :class:`.PointEval`, :class:`.SubFunction` and :class:`.Restrict` to the
postprocessor: ::

    pp.add_fields([
        PointEval("Temperature", [[0.7,0.5, 0.5]], dict(plot=True)),
        SubFunction("Temperature", slicemesh, dict(plot=True, plot_args=dict(range_min=-params.amplitude, range_max=params.amplitude, mode="color"))),
        Restrict("Temperature", submesh, dict(plot=True, save=True)),
        ])

Averages and norms
------------------------
We can also compute scalars from other fields. :class:`.DomainAvg` compute the average of a specified domain
(if not specified, the whole domain). Here, we compute the average temperature inside and outside the domain
of different diffusivity, as specified by the variable *cell_domains*: ::

    pp.add_fields([
        DomainAvg("Temperature", cell_domains=cell_domains, indicator=1, label="inner"),
        DomainAvg("Temperature", cell_domains=cell_domains, indicator=0, label="outer"),
    ])

The added parameter *label* does that these fields are now identified by *DomainAvg_Temperature-inner* and
*DomainAvg_Temperature-inner*, respectively.

We can also compute the norm of any field: ::

    pp.add_field(Norm("Temperature", dict(save=True)))

If no norm is specified, the L2-norm (or l2-norm) is computed.


Custom fields
-----------------------------
The user may also customize fields as he wishes. In this section we demonstrate two ways to compute the difference
in average temperature between the two areas of different diffusivity at any given time. First, we take an
approach based solely on accessing the *Temperature*-field: ::

    class TempDiff1(Field):
        def __init__(self, domains, ind1, ind2, *args, **kwargs):
            Field.__init__(self, *args, **kwargs)
            self.domains = domains
            self.ind1 = ind1
            self.ind2 = ind2
        
        def before_first_compute(self, get):
            self.V1 = assemble(Constant(1)*dx(self.ind1), cell_domains=self.domains, mesh=self.domains.mesh())
            self.V2 = assemble(Constant(1)*dx(self.ind2), cell_domains=self.domains, mesh=self.domains.mesh())
            
        def compute(self, get):
            u = get("Temperature")        
            T1 = 1.0/self.V1*assemble(u*dx(self.ind1), cell_domains=self.domains)
            T2 = 1.0/self.V2*assemble(u*dx(self.ind2), cell_domains=self.domains)
            return T1-T2

In this implementation we have to specify the domains, as well as compute the respective averages directly
each time. However, since we already added fields to compute the averages in both domains, there is another,
much less code-demanding way to do this: ::

    class TempDiff2(Field):
        def compute(self, get):
            T1 = get("DomainAvg_Temperature-inner")
            T2 = get("DomainAvg_Temperature-outer")
            return T1-T2

Here, we use the provided *get*-function to access the fields named as above, and compute the difference.
We add an instance of both to the potsprocessor: ::

    pp.add_fields([
        TempDiff1(cell_domains, 1, 0, dict(plot=True)),
        TempDiff2(dict(plot=True)),
    ])

Since both these should be the same, we can check this with :class:`.ErrorNorm`: ::

    pp.add_field(
        ErrorNorm("TempDiff1", "TempDiff2", dict(plot=True), name="error"),
    )

We ask for the error to be plotted. Since this is a scalar, this will be done using matplotlibs
*pyplot*-module. We also pass the keyword argument *name*, which overrides the default naming (which
would have been ErrorNorm_TempDiff1_TempDiff2) with *error*.

Combining fields
------------------------------------
Finally, we can also add combination of fields, provided all dependencies have already been added to the
postprocessor. For example, we can compute the space average of a time-average of our field
*Restrict_Temperature* the following way: ::

    pp.add_fields([
        TimeAverage("Restrict_Temperature"),
        DomainAvg("TimeAverage_Restrict_Temperature", params=dict(save=True)),
    ])

If *TimeAverage("Restrict_Temperature")* is not added first, adding the :class:`.DomainAvg`-field would
fail with a :class:`.DependencyException`, since the postprocessor would have no knowledge of the field
*TimeAverage_Restrict_Temperature*.

Saving mesh and parameters
--------------------------------------

We choose to store the mesh, domains and parameters associated with the problem: ::

    pp.store_mesh(mesh, cell_domains=cell_domains)
    pp.store_params(params)
    
These will be stored to *mesh.hdf5*, *params.pickle* and *params.txt* in the case directory.

Solving the problem
______________________________________________
Solving the problem is done very simply here using simple FEniCS-commands: ::

    solver = KrylovSolver(A, "cg", "hypre_amg")
    while t <= params.T+DOLFIN_EPS:
        cbc_print("Time: "+str(t))
        u0.t = float(t)
    
        assemble(L, tensor=b)
        bc.apply(b)
        solver.solve(U.vector(), b)
        
        # Update the postprocessor
        pp.update_all({"Temperature": lambda: U}, t, timestep)
        
        # Update time
        t += float(dt)
        timestep += 1

Note the single call to the postprocessor, *pp.update_all*, which will then execute the logic for the
postprocessor. The solution *Temperature* is passed in a dict as a lambda-function. This lambda-function
gives the user flexibility to process the solution in any way before it is used in the postprocessor. This
can for example be a scaling to physical units or joining scalar functions to a vector function.

Finally, at the end of the time-loop we finalize the postprocessor through ::

    pp.finalize_all()

This command will finalize and return values for fields such as for example time integrals.









