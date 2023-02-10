---
title: 'matscipy: Materials science with Python at the atomic-scale'
tags:
  - Python
  - Material Science
  - Atomistic simulations
authors:
  - name: Petr Grigorev
    orcid: 0000-0002-6409-9092
    #equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: "1" # (Multiple affiliations must be quoted)
  - name: Thomas Reichenbach
    orcid: 0000-0001-7477-6248
    affiliation: 4
  - name: Andreas Klemenz
    orcid: 0000-0001-5677-5639
    affiliation: 4
  - name: Gianpietro Moras
    orcid: 0000-0002-4623-2881
    affiliation: 4
  - name: Jan Grießer
    orcid: 0000-0003-2149-6730
    affiliation: 3
  - name: Jonas A. Oldenstaedt
    orcid: 
    affiliation: 3
  - name: Lucas Frérot
    orcid: 0000-0002-4138-1052
    affiliation: 3
  - name: Wolfram G. Nöhring
    orcid: 0000-0003-4203-755X
    affiliation: 3
  - name: James R. Kermode
    orcid: 0000-0001-6755-6271
    #equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 2
  - name: Lars Pastewka
    orcid: 0000-0001-8351-7336
    #equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 3
  - name: Johannes L. Hörmann
    orcid: 0000-0001-5867-695X
    affiliation: 3
affiliations:
  - name: Aix-Marseille Universit ́e, CNRS, CINaM UMR 7325, Campus de Luminy, 13288 Marseille, France
    index: 1
  - name: Warwick Centre for Predictive Modelling, School of Engineering, University of Warwick, Coventry CV4 7AL, United Kingdom
    index: 2
  - name: Department of Microsystems Engineering, University of Freiburg, 79110 Freiburg, Germany
    index: 3
  - name: Fraunhofer IWM, MicroTribology Center $\mu$TC, Wöhlerstr. 11, 79108, Freiburg, Germany
    index: 4
date: 27 October 2022
bibliography: paper.bib
---

# Summary

Behaviour of materials is governed by physical phenomena happening at an extreme range of length and time scales. Thus, computational modelling requires a multiscale approach. Simulation techniques operating on the atomic scale serve as a foundation for such an approach, providing necessary parameters for upper scale models. The physical model employed for an atomic simulation can vary from electronic structure calculations to empirical force fields, however, the set of tools for construction, manipulation and analysis of the system stays the same for a given application. Here we present a collection of such tools for applications including fracture mechanics, plasticity, contact mechanics, tribology and electrochemistry. 

# Statement of need

The python package `matscipy` contains a set of tools for researchers in a field of materials science and atomistic modelling. It is built around Atomic Simulation Environment (ASE) [@Larsen2017] that offers great flexibility and interoperability by providing a python interface to tens of simulation codes implementing different physical models. Below, we give a short summary for every application domain:

- **Plasticity.** Dislocations are extended defects in the material and are the main carriers of plasticity. `matscipy` module `dislocation.py` focuses on tools for studying structure and movement of dislocations. Construction and analysis of model atomic systems is implemented for compact and dissociated screw, as well as edge dislocations in cubic crystals. The implementation also includes large scale systems for single and double kinks. The module was employed in a study of interaction of hydrogen with screw dislocation in tungsten [@Grigorev2020].

- **Fracture mechanics.** The `matscipy.fracture_mechanics` module provides functionality for generating and applying the continuum linear elastic displacement fields near to crack tips, including support for anisotropy in cubic crystals [@Sih1965]. This functionality has been used to quantify 'lattice trapping', i.e. the effects of the discreteness of the atomic lattice on crack propagation and to compare simulations with experimental measurements of crack speeds in silicon [@Kermode2015]. There is also support for flexible boundary conditions in fracture simulations using the formalism proposed by Sinclair [@Sinclair1975] where the finite atomistic domain is coupled to an infinite elastic continuum. Finally, we provide an extension of this approach to give a numerical-continuation-enhanced flexible boundary scheme, enabling full solution paths for cracks to be computed with pseudo-arclength continuation [@Buze2021].

- **Contact mechanics**

- **Electrochemistry.** Classical continuum models describing species transport are limited in their applicability. Neither do they account for structured layering of ions in a polar solvent like water [@Seidl2021] nor do they describe finite size effects at high concentrations. Their smooth concentration distributions may, however, yield good approximations for sampling discrete particle positions to serve as initial configurations in atomistic calculations for further investigation of phenomena arising on the molecular scale. The `matscipy.electrochemistry` module provides tools that sample discrete coordinate sets from continuum distributions and apply a steric correction [@Martinez2009] to such sampled coordinate sets in order to avoid overlap of finite size species. In addition, a controlled volumes solver [@Selberherr1984] for the Poisson-Nernst-Planck equations [@Bazant2006] on a one-dimensional interval for a variety of typical boundary conditions and constraints provides a prototypical use case for the sampling and steric correction utilities.

- **Tribology.** Molecular dynamics simulations of representative volume elements of tribological interfaces are routinely used to gain insights into the atomistic mechanisms underlying friction and wear. The `matscipy` module `pressurecoupling.py` provides tools to perform such simulations under a constant normal load and sliding velocity. The module includes an implementation of the pressure coupling algorithm described in Ref. [@Pastewka2010]. By dynamically adjusting the distance between the two sliding surfaces according to the local pressure, the algorithm ensures mechanical boundary conditions that account for the inertia of the bulk material which is not explicitly included in the simulation. 

As well as these domain-specific tools, `matscipy` contains general utility functionality which is widely applicable:

- **Neighbour list**. An efficient linear-scaling neighbour list implemented in C which delivers orders-of-magnitude faster performance for large systems that the pure Python implementation in ASE [@Larsen2017]. This is becoming widely used for post-processing and structural analysis of the trajectories resulting from molecular dynamics simulations. 

- **Atomic strain**

- **Ring analysis**

- **Radial, angular and spatial correlation functions**

- **Second-order derivatives of interatomic potentials.** The thermodynamic and elastic properties of solid materials are of crucial importance for any technical application and therefore are commonly characterized in experiments and simulations. The modules `numerical` and `elasticity` provide a wide range of routines for the numerical evaluation of the dynamical matrix and the elastic properties (e.g. least-square approximation on deformed configurations, with or without energy minimization) for arbitrary interatomic potentials. In addition, `matscipy` implements analytic second-order potential derivatives for a wide range of force-fields (Pair-potentials, Bond-order potentials, Cluster potentials and Electrostatic interaction), in particular for a generic manybody potential form [@Griesser2022] in `calculators.manybody`, which encompasses a host of classical potentials [@Lennard1931;@StillingerWeber1985;@Kumagai2007;@Tersoff1989;@Brenner1990;@BKS1990]. These analytic second-order derivatives allow a fast and accurate computation of the aforementioned properties in crystals, polymers and amorphous solids, even for unstable configurations where numerical methods are not applicable. Since they are exposed as calculator properties, second-order derivatives for different force-fields can be easily combined.

# Acknowledgements

We thank Alexander Held for initial contributions to the `pressurecoupling.py` module.

`matscipy` was partially funded by the Deutsch Forschungsgemeinschaft (project 258153560) and by the Engineering and Physical Sciences Research Council (grants EP/P002188/1, EP/R012474/1 and EP/R043612/1).

# References
