# TMDSimPy: Tribomechadynamics Simulations for Python  

This repository contains python files and functions for running numerical simulations for various tribomechadynamics problems. 
Files intended for experimental analysis are not included here and can be found elsewhere. 
Specific projects are based in other repositories and use these shared modeling routines as a dependency. 
This repository is not intended to include research problems beyond a few examples.

## Reference

If using this code, please cite the relevant journal paper ([preprint here](https://doi.org/10.48550/arXiv.2401.08790)):
```
@article{porterTrackingSuperharmonic2024,
    title = {Tracking Superharmonic Resonances for Nonlinear Vibration},
    journal = {Mechanical Systems and Signal Processing},
    author = {Justin H. Porter and Matthew R.W. Brake},
    year = {Under Review},
}
```
For the rough contact friction model, please cite (MATLAB code for this model and the paper preprint can be found [here](https://github.com/tmd-lab/microslip-rough-contact)):  
```
@article{porterTowardsAPredictive2023,
    title = {Towards a predictive, physics-based friction model for the dynamics of jointed structures},
    journal = {Mechanical Systems and Signal Processing},
    volume = {192},
    pages = {110210},
    year = {2023},
    issn = {0888-3270},
    doi = {10.1016/j.ymssp.2023.110210},
    author = {Justin H. Porter and Matthew R.W. Brake},
}
```
This code is provided under the MIT License to aid in research, no guarantee is made for the accuracy of the results when applied to other structures.


## Setup 

For research analyses, it is recommended to clone this as a dependency of a different repository containing scripts that define the analyses. 
The following instructions all utilize a command line (Linux, macOS, WSL on Windows). 

This example will clone the repo and then setup a conda environment named `tmdsimpy` for the installation. 
Instructions for installing anaconda can be found [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).
 
To clone and install requirements:
```
git clone git@github.com:tmd-lab/tmdsimpy.git # clone with whatever method desired.
cd tmdsimpy
conda create --name tmdsimpy python=3.10.13 pip # create 'tmdsimpy' conda environment.
conda activate tmdsimpy
python3 -m pip install --upgrade -r requirements.txt 
```

After cloning the repo and installing the requirements, you should run these tests to ensure that everything is working.
These have been combined in a bash script, so execute:
```
source run_tests.sh
```

Note, that there is no particular reason for this specific version of python, and an attempt will be made
to keep the code compatible with current packages. 
Specific package versions are included in `specific_reqs.txt`, but may capture some unneccessary packages.
To use that list, replace the final install line from above with
```
python3 -m pip install -r specific_reqs.txt
```
You can verify the installed versions with `python3 -m pip list`.


### Setup Notes

1. This code has been developed using a x86_64 Linux machine and WSL on a Windows machine.
2. Note that JAX may not fully support all operating systems as described [here](https://jax.readthedocs.io/en/latest/installation.html). 
2. See [Windows Computer Environment](#windows-computer-environment) for more advice on working with WSL.
3. Installing packages directly through conda may be a more reliable way to get a fully reproducible environment, but can be challenging. 
   The use of conda environments to isolate code while exclusively installing via pip appears to be a 
   reasonable approach based on [this](https://www.anaconda.com/blog/using-pip-in-a-conda-environment).


## Examples

Several examples are included to demonstrate the repository and can be run to further verify the correctness of the code. 
One can also look at the tests folder to see further examples.

### Brake-Reuss Beam with Physics-Based Rough Contact

This example calculates the nonlinear vibration response of the Brake-Ruess Beam as described in [this paper](https://doi.org/10.1016/j.ymssp.2023.110210) and originally implemented in MATLAB [here](https://github.com/tmd-lab/microslip-rough-contact). This model uses provided system matrices originally calculated with Abaqus as described in [this paper](https://doi.org/10.1016/j.ymssp.2020.106615) and [this tutorial](https://nidish96.github.io/Abaqus4Joints/). Model reduction was further conducted as described in [this paper](https://doi.org/10.1016/j.ymssp.2020.107249).

This example runs continuation to calculate the modal backbone with the Extended Periodic Motion Concept (EPMC) utilizing a physics-based contact model. This example requires the full repository (and JAX). 
Therefore, this example assumes you are running on a command line (Linux, macOS, or WSL). 
Starting from the top level of the repository:
```
cd examples/structures
python3 brb_epmc.py -meso 1
```
Simulations with the default model take about 5-10 minutes on a computer with 12 cores, 32 GB of RAM and a 2.1 GHz processor. 
The command line argument `-meso 1` can be changed to `-meso 0` to run the simulation with a flat interface instead. 
While `brb_epmc.py` is running, you can look at a summary of the current results in the file `results/brb_epmc_meso_sum.dat` or `results/brb_epmc_flat_sum.dat` for with and without mesoscale topology respectively.

The results can then be checked against the published reference solution by running
```
python3 compare_brb_epmc.py -meso 1 # use same value of -meso input argument
```
For `compare_brb_epmc.py`, you may want to run this script in an IDE (e.g., spyder) for plotting instead of from the terminal (this script does not require JAX). You can change `default_mesoscale` in the script to change if it plots the comparison with or without mesoscale topology. 
Errors in the comparison are attributed to: 
1. Interpolation from different continuation points. 
2. Different mesh if using the default mesh. 
3. Slight difference in the initialization of frictional sliders (e.g., residual tractions as described [here](https://doi.org/10.1016/j.ymssp.2023.110651)). 
4. Numerical solver tolerances. 

The default model uses 122 zero-thickness elements (ZTEs). An alternative model using 232 ZTEs can be run by downloading the alternative matrices from [here](https://rice.box.com/s/y6q1fpm177mjp3ezkohrz295hhqpu3yn) to the `examples/structures/data` folder and running the lines:
```
cd examples/structures
python3 brb_epmc.py -meso 1 -system './data/BRB_ROM_U_232ELS4py.mat'
python3 compare_brb_epmc.py -meso 1 -system './data/BRB_ROM_U_232ELS4py.mat'  
```
Note, that you will need to delete any previous saved simulation results before running continuation with a different model. E.g.,
```
cd examples/structures/results
# Remove saved numpy since these will throw errors when appending new data
rm brb_epmc_meso_full.npz
rm brb_epmc_flat_full.npz
# Remove summary files since these will just get appended to with new simulations
rm brb_epmc_meso_sum.dat
rm brb_epmc_flat_sum.dat
cd ..
```

## Code Development Guidance

### Documentation

Documentation is in progress utilizing numpy docstring formatting as described [here](https://numpydoc.readthedocs.io/en/latest/format.html). 
All new functions, modules, and classes should follow correct formatting of docstrings. 
Methods, and functions should all include for "Parameters" and "Returns" sections.
All Classes should have a "Parameters" section that includes information for the constructor method.
The intention is to utilize Sphinx to generate documentation once everything is in correct numpy docstring formatting.


### Testing

New functions should have unit tests associated with them. This requires that the filename start with "test_" and that the files follow the unittest packages requirements (see existing tests for examples). 
All analytical gradients should use `tests/verification_utils.py` to check gradients against numerical approximations. 


Individual tests can also be run as files in an IDE (assuming the correct lines are included at the bottom of the file) or with the command
```
cd tests
python3 -m unittest test_epmc.py
```
A single test can be run as
```
python3 test_epmc.py TestEPMC.test_static_force
```

### JAX and JIT

The JAX library is a powerful tool for automatic differentiation and just-in-time (JIT) compilation. However, JAX has some very specific rules. One should carefully review their documentation [here](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html).


Notes on JAX implementations:
1. The JIT versions of the code assume that the same list of harmonics and the same number of time steps for AFT are used everywhere. If this is not the case, the routines will be forced to compile multiple versions for different sets of harmonics.
2. The present implementation assumes that 64-bit precision is desired. Therefore the init file sets JAX to use 64-bit. If you use JAX before importing tmdsimpy, then the correct precision may not be used. 
3. JAX/JIT examples have been created for Jenkins and Elastic Dry Friction nonlinearities (AFT only). 
It is not recommended to use the JAX versions for Jenkins since they perform worse than the vectorized Jenkins algorithm for large Nt. A non-JAX implementation of Elastic Dry Friction is not provided and future work will likely exploit JAX for auto diff to decrease development time. 
It is noted that the traditional AFT algorithm for Jenkins is much faster with JAX/JIT than traditional code.


## Acknowledgements and Funding

This material is based upon work
	supported by the U.S. Department of Energy, Office of
	Science, Office of Advanced Scientific Computing
	Research, Department of Energy Computational Science
	Graduate Fellowship under Award Number(s) DE-SC0021110.
The authors are thankful for the support of the National
Science Foundation under Grant Number 1847130.

This report was prepared as an account of
work sponsored by an agency of the United States
Government. Neither the United States Government nor
any agency thereof, nor any of their employees, makes
any warranty, express or implied, or assumes any legal
liability or responsibility for the accuracy, completeness,
or usefulness of any information, apparatus, product, or
process disclosed, or represents that its use would not
infringe privately owned rights. Reference herein to any
specific commercial product, process, or service by trade
name, trademark, manufacturer, or otherwise does not
necessarily constitute or imply its endorsement,
recommendation, or favoring by the United States
Government or any agency thereof. The views and
opinions of authors expressed herein do not necessarily
state or reflect those of the United States Government or
any agency thereof.

## Appendix

### Windows Computer Environment

If working on a Windows machine with WSL, the following workflow is recommended: 
1. Utilize a standard python IDE installed on the windows side of the computer for editing code. 
2. Run the code in the WSL terminal. For example:
   ```
   cd examples 
   python3 2dof_eldry_fric.py
   ```
3. If you need to debug code, you will need to use the 'pdb' library in python. The easiest way to start with this is to add the line
   ```
   import pdb; pdb.set_trace()
   ```
   or
   ```
   breakpoint()
   ```
   wherever you want the code to pause when you are running it.
   Make sure to save the code, then execute the code from the terminal. 
   When it pauses, it opens a python command line where you can query variables and do simple calculations to check the correctness. 
   You can use 'c' to continue the execution or 'q' to quit the execution. The commands 'l' and 'll' will show the surrounding code if you are not sure where it has paused. More information on pdb can be found [here](https://docs.python.org/3/library/pdb.html#debugger-commands).
4. If you need to generate figures, you can either have the script save them (e.g., as a .png file), or you can save the data and write a separate script to plot the results (with the later run on Windows or Linux as you desire). Using interactive plots from the terminal may be possible, but has not been tested.

