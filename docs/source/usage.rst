Usage
=====

Installation
------------

To use TMDSimPy, first clone the repository:

.. code-block:: console

   (.venv) $ git clone https://github.com/tmd-lab/tmdsimpy.git

General Use Case
----------------

Abbreviations
-------------

DOFs - Degrees of Freedom


Conventions and Common Variables
--------------------------------

U,h - list of harmonic DOFs, list of harmonics. Each global DOF is included multiple times for different harmonic components. First all of the DOFs for the first harmonic component are listed, then the next etc.
`h` should include only zero and positive integers. If interested in subharmonic motion, apply forcing at a harmonic greater than 1. Be aware that not all methods support such analyses.

Nhc - number of harmonic components. Can be calculated as `Nhc = tmdsimpy.harmonic_utils.Nhc(h)`

Derivative/Jacobian matrices - each row corresponds to the thing that the derivative is being taken of. Each column is the derivative with respect to a specific input.

Nonlinear forces are internal forces and thus positive displacements generally result in positive nonlinear forces (corresponding to resisting motion).
This is because the nonlinear forces appear on the same side of the equation as the acceleration terms.

Residual - a scalar or 1D array of values that should be converged to zero so that the equations of interest are satisfied.
