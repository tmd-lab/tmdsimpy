Usage
=====

Installation
------------

Please see the installation instructions on the GitHub Repository for cloning and setting up TMDSimPy: 
`https://github.com/tmd-lab/tmdsimpy#setup <https://github.com/tmd-lab/tmdsimpy#setup>`_

General Use Case
----------------

The typical analysis consists of the steps below. Depending on the desired quantity, many steps may be omitted.

#. Create a `VibrationSystem` object.
#. Add nonlinear forces to the vibration system.
#. (Optional) Do a static prestress analysis. 
   This generally requires solving a set of nonlinear equations with a `NonlinearSolver` object.
   You may use these results in initialization of nonlinear forces.
#. Construct an initial guess to the problem, generally based on a linearized solution.
   For some systems of equations, guesses can be constructed with `tmdsimpy.utils.harmonic.predict_harmonic_solution`.
#. Do continuation of the desired set of equations for a range of parameters using the `Continuation` class. 
   Equations are generally implemented as methods of `VibrationSystem`.
#. (Optional) Apply post processing (`tmdsimpy.postprocess`) or a reduced order model (`tmdsimpy.roms`).

Examples can be found in the repository at `https://github.com/tmd-lab/tmdsimpy/tree/develop/examples <https://github.com/tmd-lab/tmdsimpy/tree/develop/examples>`_



Abbreviations
-------------

AFT - Alternating Frequency-Time Method 

DOFs - Degrees of Freedom

EPMC - Extended Periodic Motion Concept

FFT - Fast Fourier Transform

HBM - Harmonic Balance Method

ROM - Reduced Order Model

VPRNM - Variable Phase Resonance Nonlinear Modes

Conventions and Common Variables
--------------------------------

`Nt` - Number of time instants for a cycle for frequency domain analysis.
Does not include a time instant at exactly one period because that repeats the initial point.
Should be a power of 2 for most efficient usage of FFTs.

.. code-block:: python

   Nt = 2**7 # Should be a power of 2

`tau` - nondimensional time with a cycle having a period of 1.0.

.. code-block:: python

   import numpy as np
   tau = np.linspace(0, 1, Nt+1)

`h` - should include only zero and positive integers and be sorted. If interested in subharmonic motion, apply forcing at a harmonic greater than 1. Be aware that not all methods support such analyses.

.. code-block:: python
   
   h_max = 3
   h = np.arange(h_max+1)

`Nhc` - number of harmonic components. Can be calculated as

.. code-block:: python
   
   import tmdsimpy.utils

   Nhc = tmdsimpy.utils.harmonic.Nhc(h)

`U` - list of harmonic DOFs, generally associated with a list of harmonics `h`.
Each global DOF is included multiple times for different harmonic components.
First all of the DOFs for the first harmonic component are listed, then the next etc.

.. code-block:: python

   Ndof = 3 # number of DOFs of system

   rng = np.random.default_rng(1023)
   U = rng.random(Nhc*Ndof)

   # time series over a cycle
   h0 = h[0] == 0 # bool, zeroth harmonic is included

   # time series over a cycle
   # Rows are time instants, columns are DOFs.
   x_t = np.zeros((Nt+1, Ndof))

   for ind,harmonic in enumerate(h):
       
       if harmonic == 0:
           x_t += U[:Ndof]
       else:
           x_t += U[Ndof*(2*ind - h0):Ndof*(2*ind + 1 - h0)].reshape(1, -1) \
                  * np.cos(harmonic*2*np.pi*tau).reshape(-1, 1)

           x_t += U[Ndof*(2*ind - h0 + 1):Ndof*(2*ind + 2 - h0)].reshape(1,-1) \
                  * np.sin(harmonic*2*np.pi*tau).reshape(-1, 1)

The time series can also be constructed with the `tmdsimpy` method as

.. code-block:: python

   x_t_method = tmdsimpy.utils.harmonic.time_series_deriv(Nt, h, U.reshape(Nhc, Ndof), 0)

   np.abs(x_t_method - x_t[:-1]).max()

`Uwxa` - Format of unknowns for EPMC.
This corresponds to a set of harmonic displacements the same as `U`, then a number of other variables.
Here `w = Uwxa[-3]` is the frequency in rad/s, `x = Uwxa[-2]` is the mass proportional excitation coefficient,
and `a = Uwxa[-1]` is an amplitude measure (logscale).
Other equations have similar representations.

`R` - Residual vector for a set of equations. 
Generally of size one less than the unknown vector since the unknown vector includes the continuation parameter.
The equations are solved when all entries are (near) zero.

`dRdX` - Derivative of variable `R` with respect to variable `X`.
Each row corresponds to a different entry of `R`.
Each column is the derivative with respect to a different index of `X`.
Other names may be substituted instead of `R` and `X` (e.g., nonlinear forces instead of residual).

`fnl` - nonlinear forces, these are internal forces.
Thus, positive displacements generally result in positive nonlinear forces (resisting motion).
This is because the nonlinear forces appear on the same side of the equation as the acceleration terms.

