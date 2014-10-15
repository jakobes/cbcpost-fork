from cbcpost import *
from cbcpost.utils import cbc_print
from dolfin import *
set_log_level(WARNING)

# Create parameters for problem
params = ParamDict(
    T0 = 3.0,           # Start time
    T = 6.0,            # End time
    dt = 0.05,          # Time step
    theta = 0.5,        # Time stepping scheme (0.5=Crank-Nicolson)
    alpha0 = 10.0,      # Outer diffusivity
    alpha1 = 1e-3,      # Inner diffusivity
    amplitude = 3.0,    # Amplitude of boundary condition
)

# Create mesh
mesh = UnitCubeMesh(21,21,21)

# Function spaces
V = FunctionSpace(mesh, "CG", 1)
u,v = TrialFunction(V), TestFunction(V)

# Time and time-stepping
t = params.T0
timestep = int(t/params.dt)
dt = Constant(params.dt)

restart = Restart(dict(casedir="../Basic/Results/"))
restart_data = restart.get_restart_conditions()

# Initial condition
U = restart_data.values()[0]["Temperature"]

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

# Postprocessor
pp = PostProcessor(dict(casedir="../Basic/Results", clean_casedir=False))
pp.add_field(SolutionField("Temperature", dict(save=True,
                                               save_as=["hdf5", "xdmf"],
                                               plot=True,
                                               plot_args=dict(range_min=-params.amplitude, range_max=params.amplitude),
                                               )))

from cbcpost.utils import Slice
slicemesh = Slice(mesh, (0.7,0.5,0.5), (0.0,0.0,1.0))
pp.add_fields([
    SubFunction("Temperature", slicemesh, dict(plot=True, plot_args=dict(range_min=-params.amplitude, range_max=params.amplitude, mode="color"))),
    ])

# Solve
solver = KrylovSolver(A, "cg", "hypre_amg")
while t <= params.T+DOLFIN_EPS:
    cbc_print("Time: "+str(t))
    u0.t = float(t)
    assemble(L, tensor=b)
    bc.apply(A,b)
    solver.solve(U.vector(), b)

    pp.update_all({"Temperature": lambda: U}, t, timestep)
    t += float(dt)
    timestep += 1

pp.finalize_all()
interactive()












