#!/usr/bin/env py.test

def test_imports():
    # Core
    core = ["PostProcessor", "Restart", "Replay"]
    for c in core:
        exec("from cbcpost import %s" %c)
        
    # Field bases
    field_bases = ["SolutionField", "Field", "MetaField", "MetaField2"]
    for f in field_bases:
        exec("from cbcpost import %s" %f)
    
    # Meta fields
    metafields = ["Boundary", "DomainAvg", "ErrorNorm", "Magnitude", "Maximum",
                  "Minimum", "Norm", "PointEval", "Restrict", "SubFunction", "TimeAverage",
                  "TimeDerivative", "TimeIntegral"]
    for mf in metafields:
        exec("from cbcpost import %s" %mf)
    
    # Tools
    tools = ["ParamDict", "Parameterized", "SpacePool", "get_grad_space"]
    for t in tools:
        exec("from cbcpost import %s" %t)
    
    # Modules
    modules = ["fieldbases", "meta_fields", "metafields", "planner", "plotter", "postprocessor",
               "paramdict", "parameterized", "replay", "restart", "saver", "spacepool", "utils"]
    
    # Check that imported core, field bases, meta fields and tools match entire cbcpost, excluding modules
    import cbcpost
    all_imports = dir(cbcpost)
    all_imports = [m for m in all_imports if m[0] != "_"]
    
    assert set(core+field_bases+metafields+tools) == set(all_imports)-set(modules)
