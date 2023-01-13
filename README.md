# tmd-sim-py
Tribomechadynamics Python Simulation Scripts 

This repository contains python files and functions for running numerical simulations for various tribomechadynamics problems. Files intended for experimental analysis are not included here and can be found elsewhere. Some specific analyses are provided as examples. Specific projects are based in other repositories and use these shared modeling routines as a dependency.

## Acknowledgments

In future - add details for citing the repository and/or relevant papers here.

## Usage

This repository is intended to be cloned into a repository to provide necessary functions that are used for many different modeling cases.

## Tests and Examples

All new routines added to this repository should have tests that verify that the routines give correct/expected results (add to TESTS folder). These tests should serve as good examples of how to use the related functions. Additional examples may be added to the EXAMPLES folder. 

### Test Guidelines

1. Logically name test files. 
2. Place comments at the top of files with what is being tested.
3. Clearly indicate if expected results are produced with the outputs.
4. Verify all analytical gradients numerically. 

### Test Summary for Important Functions

This section summarizes files in the TESTS folder that also serve as examples for important functions. The folder 'MATLAB_VERSIONS' contains previous implementations of functions from MATLAB that are verified against in some tests.

- *Nonlinear Forces* - see files under NL_FORCES folder.
    - *Alternating Frequency Time (AFT)* see verify_aft.py (Duffing) and verify_hysteretic_aft.py (Jenkins)
- *Continuation* - see verify_continuation.py
- *Extended Periodic Motion Concept (EPMC)* - see verify_epmc.py
- *Harmonic Balance Method (HBM)* - see verify_hbm.py
    - *HBM Utilities* - verify_harmonic_utils.py
- *Nonlinear Solvers* - verify_solver.py


