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

1. Logically name test files. The filename must start with "test_" for the unittest framework.  
2. Place comments at the top of files with what is being tested.
3. Clearly indicate if expected results are produced with the outputs.
4. Verify all analytical gradients numerically, use functions in 'verification_utils.py'. 

#### Unittest Framework 

All tests should use the unittest framework. This requires that all tests start with the word "test" in the filename. This allows for easier running of tests and integration with existing tools for continuous testing. Unittest also requires that values be checked using class assertion statements (e.g., self.assertLess()).

Tests written in the unittest framework can be run by navigating to the TESTS folder and running
```
python -m unittest discover
```
In addition, this command must also be run in the TESTS/NL_FORCES folder to test the nonlinear forces.
Individual tests can also be run as files in an IDE (assuming the correct lines are included at the bottom of the file) or with the command
```
python -m unittest test_hbm_base.py
```

### Test Summary for Important Functions

This section summarizes files in the TESTS folder that also serve as examples for important functions. The folder 'MATLAB_VERSIONS' contains previous implementations of functions from MATLAB that are verified against in some tests.

- *Nonlinear Forces* - see files under NL_FORCES folder.
    - *Alternating Frequency Time (AFT)* - see test_duffing_aft.py (Duffing) and test_jenkins_hysteretic_aft.py (Jenkins)
    - The vector versions of Iwan and Jenkins are much faster than the normal version under some conditions. See EXAMPLES/vectorized_jenkins_iwan_aft.py
- *Continuation* - see test_continuation.py - uses harmonic balance and duffing.
- *Extended Periodic Motion Concept (EPMC)* - see test_epmc.py - uses continuation, Duffing, and Jenkins as well.
- *Harmonic Balance Method (HBM)* - see test_hbm.py - uses MATLAB/python integration to verify against previous routines. There is a flag at the top that can be set to False to avoid the MATLAB calls so the test can be run without the MATLAB comparisons. This function also uses the solver to check a number of solutions. 
    - *HBM Utilities* - test_harmonic_utils.py - requires MATLAB/python integration to verify against previous routines.
    - *HBM Base Excitation* - test_hbm_base.py - tests the base excitation HBM implementation.
- *Nonlinear Solvers* - test_solver.py. More detailed uses can be found with continuation. 


