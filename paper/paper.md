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
  - name: Lucas Frérot
    orcid: 0000-0002-4138-1052
    affiliation: 3
  - name: Jan Grießer
    orcid: 0000-0003-2149-6730
    affiliation: 3
  - name: Johannes L. Hörmann
    orcid: 0000-0001-5867-695X
    affiliation: 3
  - name: Andreas Klemenz
    orcid: 0000-0001-5677-5639
    affiliation: 4
  - name: Gianpietro Moras
    orcid: 0000-0002-4623-2881
    affiliation: 4
  - name: Wolfram G. Nöhring
    orcid: 0000-0003-4203-755X
    affiliation: 3
  - name: Jonas A. Oldenstaedt
    orcid: 
    affiliation: 3
  - name: Lars Pastewka
    orcid: 0000-0001-8351-7336
    #equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 3
  - name: Thomas Reichenbach
    orcid: 0000-0001-7477-6248
    affiliation: 4
  - name: James R. Kermode
    orcid: 0000-0001-6755-6271
    #equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 2
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

# TODOs

- The paper appears to split into application domains, calculators and generic tools. Should we split the repo similarly, i.e. have `matscipy.domain.fracture_mechanics`, `matscipy.tools.neighbours` and `matscipy.calculators.manybody`

- We need to clear up naming schemes, e.g. `pressurecoupling` should be `pressure_coupling`. We should also check for CamelCase vs snake_case

- LP: Add own papers to tribology section that used matscipy

- LP: I don't understand what `matscipy.opls` actually does (except for IO), this should be explained

- LP: Text for `committee` and `mcfm` calculators [Drafts added by JRK]

# Statement of need

The Python package `matscipy` contains a set of tools for researchers using atomic-scale models in materials science. In atomic-scale modelling, the primary numerical object is a point in three-dimensional space that represents the position of an individual atom. Simulations are often dynamical, where configurations change over time and each atom carries a velocity. Complexity emerges from the interactions of many atoms, and numerical tools are required for generating initial atomic configurations and for analyzing output of such dynamical simulations, most commonly to connect local geometric arrangements of atoms to physical processes. An example, also found below, is the detection of the tip of a crack that moves through a solid body.

We never see individual atoms at macroscopic scales. To understand the behaviour of everyday objects, atomic-scale information needs to be transferred to the continuum scale. This is the primary objective of multi-scale modelling. `matscipy` focuses on atomic representations of materials, but implements tools for connecting to continuum description in mechanics and transport theory. Each of the application domains described in the following therefore relies on the computation of continuum fields, that is realized through analytic or numerical solutions.

There is no other package that fills the particular niche described by the application domains in the next section that we are aware of. The package addresses the boundary between atomic-scale and continuum modelling in materials with particular emphasis on plasticity, fracture and tribology. In addition, we target interoperability with the widely used Atomic Simulation Environment (ASE), described in more detail in the code philosophy section below.

Generic multi-scale coupling packages exist. Examples of open-source codes are `libmultiscale` [@Anciaux2007;@Anciaux2018], `MultiBench` [@Miller2009] and Green's function molecular dynamics (`GFMD`) [@Campana2006;@Pastewka2012]. These packages target two-way coupling of elastic response between atomic-scale and continuum formulations, while `matscipy` focuses on constructing atomic domains from continuum information and extracting continuum field from atomic structures. `matscipy` has only limited capabilities for two-way coupling, which makes it significantly easier to use and reduces the barrier to engaging in multi-scale modelling of materials.

# Application domains

Within materials science, the package has different application domains:

- **Elasticity.** Solids respond to small external loads with reversible deformation, known as elasticity. The strength of the response is characterized by the elastic moduli. `matscipy.elasticity` implements functions for computing elastic moduli from small deformation of atomistic systems that consider potential symmetries of the underlying atomic system, in particular for crystals. `matscipy` also implements analytic calculation of elastic moduli for some interatomic potentials, described in more details below. The computation of elastic moduli is a basic prerequisite for multi-scale modelling of materials, as they are the most basic parameters of continuum material models. `matscipy` was used to study finite-pressure elastic constants and structural stability in crystals [@Griesser2023crystal] and glasses [@Griesser2023glass].

- **Plasticity.** For large loads, solids can respond with irreversible deformation. One form of irreversibility is plasticity, that is carried by extended defects, the dislocations, in crystals. The module `matscipy.dislocation` implements tools for studying structure and movement of dislocations. Construction and analysis of model atomic systems is implemented for compact and dissociated screw, as well as edge dislocations in cubic crystals. The implementation supports ideal straight as well as kinked dislocations.
<!-- LP,LF: Prior sentence is unclear: Does it include data files are functions to do large scale systems? What is special about large scale systems? PG: is it more clear now? The main message was that there are methods to study kinks as well -->
The module was employed in a study of interaction of hydrogen with screw dislocations in tungsten [@Grigorev2020].

- **Fracture mechanics.** Cracking is the process of generating new surface area by splitting the material apart. The module `matscipy.fracture_mechanics` provides functionality for calculating continuum linear elastic displacement fields near crack tips, including support for anisotropy in the elastic response [@Sih1965]. The module also implements generation of atomic structures that are deformed according to this near-tip field. This functionality has been used to quantify lattice trapping, which is the pinning of cracks due to the discreteness of the atomic lattice, and to compare simulations with experimental measurements of crack speeds in silicon [@Kermode2015]. There is also support for flexible boundary conditions in fracture simulations using the formalism proposed by Sinclair [@Sinclair1975] where the finite atomistic domain is coupled to an infinite elastic continuum. Finally, we provide an extension of this approach to give a numerical-continuation-enhanced flexible boundary scheme <!-- LP,LF: Think about prior term --> , enabling full solution paths for cracks to be computed with pseudo-arclength continuation [@Buze2021].

- **Tribology.** Tribology is the study of two comoving interfaces, such as encountered in frictional sliding or adhesion. Molecular dynamics simulations of representative volume elements of tribological interfaces are routinely used to gain insights into the atomistic mechanisms underlying friction and wear. The module `matscipy.pressurecoupling` provides tools to perform such simulations under a constant normal load and sliding velocity. It includes an implementation of the pressure coupling algorithm described by @Pastewka2010. By dynamically adjusting the distance between the two sliding surfaces according to the local pressure, the algorithm ensures mechanical boundary conditions that account for the inertia of the bulk material which is not explicitly included in the simulation. This algorithm was used to study friction XXX and wear YYY.

- **Electrochemistry.** Electrochemistry describes the motion and spatial distribution of charged atoms and molecules (ions) within an external electric field. Classical treatment of charged systems leads to continuous field that describe mean charge distributions, while true atomic systems consist of discrete particles with fixed charges. The `matscipy.electrochemistry` module provides tools that statistically sample discrete coordinate sets from continuum fields and apply steric corrections [@Martinez2009] to avoid overlap of finite size species. To generate continuum charge distributions, the package also contains a control-volume solver [@Selberherr1984] for the one-dimensional Poisson--Nernst--Planck equations [@Bazant2006], as well as an interface to the finite-element solver `FEniCS` [@LoggMardalEtAl2012]. This was used to study electrochemical control of frictional interfaces [@Seidl2021].

<!-- Classical continuum models describing species transport are limited in their applicability. Neither do they account for structured layering of ions in a polar solvent like water [@Seidl2021] nor do they describe finite size effects at high concentrations. Their smooth concentration distributions may, however, yield good approximations for sampling discrete particle positions to serve as initial configurations in atomistic calculations for further investigation of phenomena arising on the molecular scale. -->

# All-purpose atomic analysis tools

As well as these domain-specific tools, `matscipy` contains general utility functionality which is widely applicable:

- **Neighbour list.** An efficient linear-scaling neighbour list implemented in
  C delivers orders-of-magnitude faster performance for large systems than
  the pure Python implementation in ASE [@Larsen2017], see \autoref{fig:nl_time}.
  The neighbour list stored in a data structure comparable to coordinate (`COO`) sparse matrix storage format [@Saad1990],
  where two arrays contain the indices of the neighbouring atoms and further arrays store
  distance vectors, absolute distances, and other properties associated with an atomic pair.
  This allows compact code for evaluating properties that depend on pairs, such as pair-distribution function or interatomic potential energies and forces. Most of the tools described in the following rely on this neighbour list format.
  The neighbour list is becoming widely used for post-processing and structural analysis of the trajectories resulting from molecular dynamics simulations.  <!-- LF,LP: Do we need a reference for this or some evidence how widely used? JRK: now being used by MACE (https://github.com/ACEsuit/mace) and other neural network MLIPs too, could cite or link -->

![Neighbor list computation time comparison between ASE and Matscipy implementations.\label{fig:nl_time}](nl_time.svg)

- **Atomic strain.** Continuum mechanics is formulated in terms of strains, which characterizes the fractional shape changes of small volumes. Strains are typically only well defined if averaged over sufficiently large volumes, and extracting strain fields from atomic-scale calculations is notoriously difficult. `matscipy` implements calculations of strain by observing changes in local atomic neighbourhoods across trajectories. It fits a per-atom displacement gradient that minimizes the error in displacement between two configurations as described by @Falk1998. The error resulting from this fit quantifies the non-affine contribution ot the overall displacement and is known as $D^2_\text{min}$. We used this analysis to quantify local strain in the deformation of crystals [@Gola2019;@Gola2020] and glasses [@Jana2019]. <!-- LP: James, is fitting to local tetrahedral environments in matscipy? JRK: no, this was never ported out of QUIP -->

- **Radial, spatial and angular correlation functions.** Topological order in atomic-scale systems is often characterized by statistical measures of the local atomic-environment. The simplest one is the pair-distribution or radial-distribution function, that gives the probability $g_2(r)$ of finding an atom at distance $r$. For three atoms, we can define a probability of finding a specific angle, yielding the angular correlation functions. `matscipy` has utility function for computing these correlation functions to large distances, including the correlation of arbitrary additional per-atom properties such as per-atom strains.

- **Ring analysis.** Topological order in network glasses can be characterized by statistics of shortest-path rings [@Franzblau1991]. `matscipy` implements calculations of these rings using a backtracking algorithm in C. We regularly use `matscipy` to charactize shortest-path rings in amorphous carbon [@Pastewka2008;@Jana2019].



# Interatomic potentials

Besides generating and analysing atomic-scale configurations, `matscipy` implements specific interatomic potentials [@Muser2023]. The goal here is not to provide the most efficient implementation of computing interatomic forces. We rather aim to provide simple implementations for testing new functional forms, or testing new features such as the computation of derivatives of order $2$ or higher.

- **Interatomic potentials.** The module `matscipy.calculators` has implementations of classical pair-potentials, Coulomb interactions, the embedded-atom method and other many-body potentials (@Tersoff1989, @StillingerWeber1985, and others).

- **Second-order derivatives.** The thermodynamic and elastic properties of solid materials are closely connected to the Hessian of the overall system, that contains the second derivatives of the total energy with respect to position and macroscopic strains. `matscipy` implements analytic second-order potential derivatives for pair-potentials [@Lennard1931], bond-order potentials [@Kumagai2007;@Tersoff1989;@Brenner1990], cluster potentials [@StillingerWeber1985] and electrostatic interaction [@BKS1990].
This is achieved throuhg a generic mathematical formulation of the manybody total energy [@Griesser2023b] in `calculators.manybody`.
The modules `matscipy.numerical` additionally provides routines for the numerical (finite-differences) evaluation of these properties. These analytic second-order derivatives allow a fast and accurate computation of the aforementioned properties in crystals, polymers and amorphous solids, even for unstable configurations where numerical methods are not applicable.

- **Non-reactive force fields** Non-reactive force fields for molecular dynamics simulations consist of non-bonded and bonded interaction terms [@Jorgensen1996]. The latter describe the structure of molecules and solids through lists of bonds, angles, and dihedrals. The construction of these force fields is mathematically simple but requires a considerable bookkeeping effort. The module `matscipy.opls` provides efficient tools for this purpose. Input and output routines for reading and writing the corresponding control files for LAMMPS [@Thompson2022] are implemented in the module `matscipy.io.opls`. We have used this approach in various studies on tribology, wetting and nanoscale rheology [@Mayrhofer2016, @Falk2020, @Reichenbach2020, @vonGoeldel2021, @Falk2022].

- **Quantum mechanics/molecular mechanics** (QM/MM) The module `matscipy.calculators.mcfm` implements a generalised force-mixing potential [@Bernstein2009] with support for multiple concurrent QM clusters, named MultiClusterForceMixing (MCFM). It has been applied to model failure of graphene-nanotube composites [@Golebiowski2018, @Golebiowski2020].

- **Committee models**. The module `matscipy.calculators.committee` provides support for committees of interatomic potentials with the same functional form but differing parameters, in order to allow the effect of the uncertainty in parameters on model predictions to be estimated. This is typically used with machine learning interatomic potentials (MLIPs). The implementation follows the approach of [@Musil2019] where the ensemble of models is generated by training models on different subsets of a common overall training database.

# Code philosophy and structure

`matscipy` is built on the Atomic Simulation Environment (ASE) [@Larsen2017] that offers great flexibility and interoperability by providing a Python interface to tens of simulation codes implementing different physical models. While ASE's policy is to remain pure Python, we augment some of ASE's functionality with more efficient implementations in C, such as the neighbour list.

The central class in ASE is the `Atoms` class that stores atomic positions, velocities and other properties. `Calculator`s describe relationships between atoms, such as computing energies and forces, and can be attached to `Atoms` objects. All other `matscipy` functionality is implemented as functions acting on `Atoms` objects.

# Acknowledgements

We thank Alexander Held for initial contributions to the `pressurecoupling.py` module and Michael Walter for initial contributions to the `opls.py` module. `matscipy` was partially funded by the Deutsche Forschungsgemeinschaft (projects 258153560, 390951807 and 461911253), the European Research Council (ERC StG 757343), the Engineering and Physical Sciences Research Council (grants EP/P002188/1, EP/R012474/1 and EP/R043612/1) and the Leverhulme Trust under grant RPG-2017-191.

# References
