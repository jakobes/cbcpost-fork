def test_imports():
    # Core modules
    from cbcpost import PostProcessor
    from cbcpost import Restart
    from cbcpost import Replay
    
    # Field bases
    from cbcpost import Field
    from cbcpost import MetaField
    from cbcpost import MetaField2
    
    # Metafields
    from cbcpost import Boundary
    from cbcpost import BoundaryAvg
    from cbcpost import DomainAvg
    from cbcpost import ErrorNorm
    from cbcpost import Magnitude
    from cbcpost import Maximum
    from cbcpost import Minimum
    from cbcpost import Norm
    from cbcpost import PointEval
    from cbcpost import Restrict
    from cbcpost import RunningAvg
    from cbcpost import RunningL2norm
    from cbcpost import RunningMax
    from cbcpost import RunningMin
    from cbcpost import SecondTimeDerivative
    from cbcpost import SubFunction
    from cbcpost import TimeAverage
    from cbcpost import TimeDerivative
    from cbcpost import TimeIntegral

    # Tools
    from cbcpost import ParamDict
    from cbcpost import Parameterized
    from cbcpost import SpacePool
    from cbcpost import get_grad_space
    
