"""
# For a more complex problem, consider the 3D-case of the heat equation with variable diffusivity.
# 
# The general heat equation reads
# 
# ..  math::
#     \frac{\partial u}{\partial t} + \alpha \delta u = f
# 
# where u typically denotes the temperature and :math:'\alpha' denotes the material diffusivity.
# 
# Boundary conditions are in our example given as
# 
# .. math::
#     u(x,t) = Asin(2\pi t), x \in \partial \Omega
#     
# and initial condition
# 
# .. math::
#     u(x,0) = 0.
#     
# We solve the equations at the unit cube for :math:'t \in (0,5]'.
# 
# First, we set up the problem using FEniCS:
#
# """

from cbcpost import *
from cbcpost.utils import cbc_print
from dolfin import *
set_log_level(WARNING)


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

# Create mesh
mesh = UnitCubeMesh(21,21,21)

# Function spaces
V = FunctionSpace(mesh, "CG", 1)
u,v = TrialFunction(V), TestFunction(V)
U = Function(V)

# Create parameters for problem
params = ParamDict(
    T = 5.0,           # End time
    dt = 0.05,          # Time step
    theta = 0.5,       # Time stepping scheme (0.5=Crank-Nicolson)
    alpha0 = 10.0,      # Outer diffusivity
    alpha1 = 1e-3,      # Inner diffusivity
    amplitude = 3.0,    # Amplitude of boundary condition
)

# Time and time-stepping
t = 0.0
timestep = 0
dt = Constant(params.dt)

# Conductivity
alpha = project(Alpha(params.alpha0, params.alpha1), V)

# Boundary condition
u0 = Expression("ampl*sin(2*pi*t)", t=t, ampl=params.amplitude)
bc = DirichletBC(V, u0, "x[0] > 1-DOLFIN_EPS")

# Bilinear form
a = 1.0/dt*inner(u,v)*dx() + Constant(params.theta)*alpha*inner(grad(u), grad(v))*dx() #+ alpha*inner(dot(grad(u),n), v)*ds()# + b*inner(u,v)*dx()
L = 1.0/dt*inner(U,v)*dx() + Constant(1-params.theta)*alpha*inner(grad(U), grad(v))*dx()
A = assemble(a)
b = assemble(L)
bc.apply(A)


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
        
class TempDiff2(Field):
    def compute(self, get):
        T1 = get("DomainAvg_Temperature-inner")
        T2 = get("DomainAvg_Temperature-outer")
        return T1-T2


# Postprocessor
pp = PostProcessor(dict(casedir="Results", clean_casedir=True))
pp.add_field(SolutionField("Temperature", dict(save=True,
                                               save_as=["hdf5", "xdmf"],
                                               plot=True,
                                               plot_args=dict(range_min=-params.amplitude, range_max=params.amplitude),
                                               )))

# Derivatives and integrals
pp.add_fields([
    TimeIntegral("Temperature", dict(save=True, start_time=1.0, end_time=2.0)),
    TimeAverage("Temperature", dict(save=True, end_time=params.T)),
    TimeDerivative("Temperature", dict(save=True, start_time=1.0, end_time=2.0)),
    ])

# Inspect part of solution
from cbcpost.utils import create_submesh, Slice
celldomains = CellFunction("size_t", mesh)
celldomains.set_all(0)
AutoSubDomain(inside).mark(celldomains, 1)

slicemesh = Slice(mesh, (0.7,0.5,0.5), (0.0,0.0,1.0))
submesh = create_submesh(mesh, celldomains, 1)

pp.add_fields([
    PointEval("Temperature", [[0.7,0.5, 0.5]], dict(plot=True)),
    SubFunction("Temperature", slicemesh, dict(plot=True, plot_args=dict(range_min=-params.amplitude, range_max=params.amplitude, mode="color"))),
    Restrict("Temperature", submesh, dict(plot=True, save=True)),
    ])

# Averages
pp.add_fields([
    DomainAvg("Temperature", cell_domains=celldomains, indicator=1, label="inner"),
    DomainAvg("Temperature", cell_domains=celldomains, indicator=0, label="outer"),
])

# Custom fields
pp.add_fields([
    TempDiff1(celldomains, 1, 0, dict(plot=True)),
    TempDiff2(dict(plot=True)),
])

# Norm inspection
pp.add_field(
    ErrorNorm("TempDiff1", "TempDiff2", dict(plot=True), name="error"),
)

# Combination fields
pp.add_fields([
    TimeAverage("Restrict_Temperature"),
    DomainAvg("TimeAverage_Restrict_Temperature", params=dict(save=True)),
])


pp.store_mesh(mesh, cell_domains=celldomains)
pp.store_params(params)

# Solve
solver = KrylovSolver(A, "cg", "hypre_amg")
while t <= params.T+DOLFIN_EPS:
    cbc_print("Time: "+str(t))
    u0.t = float(t)
    assemble(L, tensor=b)
    bc.apply(b)
    solver.solve(U.vector(), b)
    #plot(U, range_min=-amplitude, range_max=amplitude)
    pp.update_all({"Temperature": lambda: U}, t, timestep)
    t += float(dt)
    timestep += 1

pp.finalize_all()
interactive()
    
    







    
    
    
