Command line interface
======================

`matscipy` provides command line interfaces (CLIs) for selected functionality.
These CLIs are installed per default and are prefixed with `matscipy-`.

Many of the CLI commands use a parameter file for control. This is a Python file called `params.py` that resides in
directory where the command is executed. The parameter file can be queried programmatically using the
:meth:`matscipy.has_parameter` and :meth:`matscipy.parameter` utility functions.
These functions automatically echo the values of the parameters, including whether it was specified by the user or
whether a default parameter was used, to the log.

.. toctree::

   diffusion
   electrochemistry
   fracture_mechanics
   calculators
   structure