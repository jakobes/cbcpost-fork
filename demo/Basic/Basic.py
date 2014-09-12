from cbcpost import *
from dolfin import *
set_log_level(WARNING)

D = 2
if D == 2:
    mesh = UnitSquareMesh(128,128)
elif D == 3:
    mesh = UnitCubeMesh(12,12,12)
else:
    raise RuntimeError("Dimension must be 2 or 3.")


class Alpha(Expression):
    "Variable conductivity expression"
    def __init__(self, alpha0, alpha1):
        self.alpha0 = alpha0
        self.alpha1 = alpha1
    
    def eval(self, value, x):
        d = len(x)
        if d == 2 and (0.6 < x[0] < 0.8) and (0.3 < x[1] < 0.6):
            value[0] = self.alpha1
        elif d == 3 and (0.6 < x[0] < 0.8) and (0.3 < x[1] < 0.6) and (0.2 < x[2] < 0.7):
            value[0] = self.alpha1
        else:
            value[0] = self.alpha0

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
    alpha1 = 1e-1,      # Inner diffusivity
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


# Postprocessor
pp = PostProcessor(dict(casedir="Results", clean_casedir=True))
pp.add_field(SolutionField("Temperature", dict(save=True,
                                               save_as=["hdf5", "xdmf"],
                                               plot=True,
                                               plot_args=dict(range_min=-params.amplitude, range_max=params.amplitude),
                                               )))

pp.add_fields([
    TimeDerivative("Temperature", dict(plot=True)),
    DomainAvg("Temperature", dict(plot=True)),
    TimeAverage("Temperature", dict(save=True, plot=True, start_time=2.0, end_time=4.0)),
    ])


pp.store_mesh(mesh)
pp.store_params(params)


# Solve
solver = KrylovSolver(A, "cg", "hypre_amg")
while t <= params.T+DOLFIN_EPS:
    print "Time: ", t
    u0.t = float(t)
    assemble(L, tensor=b)
    bc.apply(b)
    solver.solve(U.vector(), b)
    #plot(U, range_min=-amplitude, range_max=amplitude)
    pp.update_all({"Temperature": lambda: U}, t, timestep)
    t += float(dt)
    timestep += 1
print norm(U)
    
    







    
    
    
