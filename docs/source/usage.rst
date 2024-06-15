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

Nhc - number of harmonic components. Can be calculated as `Nhc = tmdsimpy.harmonic_utils.Nhc(h)`
