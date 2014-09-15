from cbcpost import *
from dolfin import set_log_level, WARNING, interactive
set_log_level(WARNING)

pp = PostProcessor(dict(casedir="../Basic/Results"))

pp.add_fields([
    SolutionField("Temperature", dict(plot=True)),
    Norm("Temperature", dict(save=True, plot=True)),
    TimeIntegral("Norm_Temperature", dict(save=True, start_time=0.0, end_time=6.0)),
])

replayer = Replay(pp)
replayer.replay()
interactive()