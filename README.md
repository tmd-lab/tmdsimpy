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
4. Verify all analytical gradients numerically, use functions in 'verification_utils.py'. 

#### Ongoing Work

New tests should follow the example of "test_hbm_base.py" and use the unittest framework. These tests should start with the word test for the filename. This framework allows for easier running of all tests and integration with existing tools for continuous testing.

Tests written in the unittest framework can all be run by navigating to the TESTS folder and running
```
python -m unittest discover
```
individual tests can also be run as files in an IDE or with the command
```
python -m unittest test_hbm_base.py
```

### Test Summary for Important Functions

This section summarizes files in the TESTS folder that also serve as examples for important functions. The folder 'MATLAB_VERSIONS' contains previous implementations of functions from MATLAB that are verified against in some tests.

- *Nonlinear Forces* - see files under NL_FORCES folder.
    - *Alternating Frequency Time (AFT)* - see verify_aft.py (Duffing) and verify_hysteretic_aft.py (Jenkins)
    - The vector versions of Iwan and Jenkins are much faster than the normal version under some conditions.
- *Continuation* - see test_continuation.py - uses harmonic balance and duffing.
- *Extended Periodic Motion Concept (EPMC)* - see verify_epmc.py - uses continuation, Duffing, and Jenkins as well.
- *Harmonic Balance Method (HBM)* - see verify_hbm.py - uses MATLAB/python integration to verify against previous routines. There is a flag at the top that can be set to False to avoid the MATLAB calls so the test can be run without the MATLAB comparisons. This function also uses the solver to check a number of solutions. 
    - *HBM Utilities* - verify_harmonic_utils.py - requires MATLAB/python integration to verify against previous routines.
    - *HBM Base Excitation* - test_hbm_base.py - tests the base excitation HBM implementation.
- *Nonlinear Solvers* - verify_solver.py. More detailed uses can be found with continuation. 


