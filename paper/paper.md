---
title: 'cbcpost - a postprocessing framework for FEniCS'
tags:
  - postprocessing
  - fenics
  - python
authors:
 - name: Øyvind Evju
   orcid: 0000-0002-6153-5258
   affiliation: Simula Research Laboratory
 - name: Martin Sandve Alnæs
   orcid: 0000-0001-5479-5855
   affiliation: Simula Research Laboratory
date: 4 August 2016
bibliography: paper.bib
---

# Summary

cbcpost is developed to simplify the postprocessing of simulation results, produced by FEniCS solvers [@fenics].

The framework is designed to take any given solution, and compute and save any derived data. Derived data can easily be made highly complex, due to the modular design and implementation of computations of quantities such as integrals, derivatives, magnitude etc, and the ability to chain these.

The interface is designed to be simple, with minimal cluttering of a typical solver code. This is illustrated by the following simple example:

```python
# ... problem set up ...

# Set up postprocessor
solution = SolutionField("Displacement", dict(save=True))
postprocessor = PostProcessor(dict(casedir="Results/"))
postprocessor.add_field(solution)

# Add derived fields
postprocessor.add_fields([
    Maximum("Displacement", dict(save=True)),
    TimeAverage("Displacement", dict(save=True, start_time=1.0, end_time=2.0)),
])

t = 0.0
timestep = 0
while t < T:
    timestep += 1
    # ... solve equation ...

    # Update postprocessor
    postprocessor.update_all(dict("Displacement"=lambda: u), timestep, t)

    # continue
```
# References
