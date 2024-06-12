"""
Verify the Mindlin-Iwan Fit (MIF) Model for tangential asperity contact for the
rough contact model option

Test Cases
----------

Force displacement relationships tested correspond to 
    1. Fig. 5(a) of Ref [1]_ - constant normal force cyclic force
    2. Fig. 5(b) of Ref [1]_ - decreasing normal force with increasing tangent 
        force
    3. Fig. 5(c) of Ref [1]_ - increasing normal and tangential force
    4. Check the static force calculation
    5. Check that AFT correctly hooks in with local force history
    
For all three cases, reference data is produced by the MATLAB script 
[here](https://github.com/tmd-lab/microslip-rough-contact/blob/main/PLOTS/mindlin_compare_plot.m)
corresponding to the previous MATLAB work in the initial development of the 
rough contact model. Note that 'N_test=20' should be used to reproduce
the test data for this file.

Tests should be run with two identical asperities contributing equally to load
and should be verified to be different than two different asperities contributing
equal (check vectorization etc this way)


References
----------

    [1] J. H. Porter and M. R. W. Brake, 2023, Towards a predictive, physics-
    based friction model for the dynamics of jointed structures, Mechanical 
    Systems and Signal Processing

"""

# Standard imports
import numpy as np
import sys
import unittest

# Python Utilities
sys.path.append('../..')

from tmdsimpy.jax.nlforces.roughcontact.rough_contact import RoughContactFriction


sys.path.append('..')
import verification_utils as vutils

# Import for the reference data
import yaml

###############################################################################
###     Testing Class                                                       ###
###############################################################################


class TestRoughContactMIF(unittest.TestCase):
    
    def __init__(self, *args, **kwargs):
        """
        Define tolerances here for all the tests

        Returns
        -------
        None.

        """

        super(TestRoughContactMIF, self).__init__(*args, **kwargs)     
        
        # Tolerances
        self.rel_force_tol = 1e-6 # Error on force evaluation
        self.atol_grad = 1e-16
        self.rtol_grad = 1e-8
        
        ###############
        # Load Reference Data from old Rough Contact Model
        yaml_file = './reference/mif_tangential_asperity.yaml'
        
        with open(yaml_file, 'r') as file:
            ref_dict = yaml.safe_load(file)
            
        # Construct New Rough Contact Models to Match the old ones
        Q = np.eye(3)
        T = np.eye(3)
        
        correct_model1 = RoughContactFriction(Q, T, 
                                          ref_dict['E'], 
                                          ref_dict['nu'], 
                                          ref_dict['R'], 
                                          ref_dict['Et'], 
                                          np.float64(ref_dict['Sys']), 
                                          mu=ref_dict['mu'], 
                                          u0=0, 
                                          meso_gap=0, 
                                          gaps=np.array([0.0, 0.0]), 
                                          gap_weights=np.array([0.75, 0.25]),
                                          tangent_model='MIF')
            
        correct_model2 = RoughContactFriction(Q, T, 
                                          ref_dict['E'], 
                                          ref_dict['nu'], 
                                          ref_dict['R'], 
                                          ref_dict['Et'], 
                                          np.float64(ref_dict['Sys']), 
                                          mu=ref_dict['mu'], 
                                          u0=0, 
                                          meso_gap=0, 
                                          gaps=np.array([0.0, -0.1]), 
                                          gap_weights=np.array([1.0, 0.0]),
                                          tangent_model='MIF')
        
        correct_model3 = RoughContactFriction(Q, T, 
                                          ref_dict['E'], 
                                          ref_dict['nu'], 
                                          ref_dict['R'], 
                                          ref_dict['Et'], 
                                          np.float64(ref_dict['Sys']), 
                                          mu=ref_dict['mu'], 
                                          u0=0, 
                                          meso_gap=0, 
                                          gaps=np.array([-0.1, 0.0]), 
                                          gap_weights=np.array([0.0, 1.0]),
                                          tangent_model='MIF')
        
        diff_model = RoughContactFriction(Q, T, 
                                          ref_dict['E'], 
                                          ref_dict['nu'], 
                                          ref_dict['R'], 
                                          ref_dict['Et'], 
                                          np.float64(ref_dict['Sys']), 
                                          mu=ref_dict['mu'], 
                                          u0=0, 
                                          meso_gap=0, 
                                          gaps=np.array([0.0, -0.1]), 
                                          gap_weights=np.array([0.75, 0.25]),
                                          tangent_model='MIF')
            
        self.model_correct_asps = [correct_model1, 
                                   correct_model2, 
                                   correct_model3]
        
        self.model_diff_asps = diff_model
        
        self.ref_dict = ref_dict


    def test_constant_normal_asperity(self):
        """
        Test a series of applied displacements for tangential MIF model
        at constant normal load.
        """
        
        unlth0 = np.array([0.0, 0.0, 0.0])
        
        reference = self.ref_dict['constant_normal']
        
        unlt = np.vstack((np.asarray(reference['x_disp']), 
                          0*np.asarray(reference['x_disp']),
                          np.asarray(reference['normal_disp']))).T
        
        for ind, model in enumerate(self.model_correct_asps):
            
            fxyn_t = model.local_force_history(unlt, 0*unlt, 0, 0, unlth0, 
                                               max_repeats=1)[0]
            
            ft_err = np.linalg.norm(fxyn_t[:, 0] 
                                    - np.asarray(reference['x_force'])) \
                        / np.linalg.norm(reference['x_force'])
            
            self.assertLess(ft_err, self.rel_force_tol,
                            'Tangent force is incorrect for constant load and '
                            + 'model {}.'.format(ind))
            
            fn_err = np.linalg.norm(fxyn_t[:, 2] 
                                    - np.asarray(reference['normal_force'])) \
                        / reference['normal_force'] / np.sqrt(fxyn_t.shape[0])
            
            self.assertLess(fn_err, self.rel_force_tol,
                            'Normal force is incorrect for constant load and '
                            + 'model {}.'.format(ind))
            
        # Check that changing the integration scheme correctly results in 
        # different results
        model = self.model_diff_asps
        
        fxyn_t = model.local_force_history(unlt, 0*unlt, 0, 0, unlth0, 
                                           max_repeats=1)[0]
        
        ft_err = np.linalg.norm(fxyn_t[:, 0] 
                                - np.asarray(reference['x_force'])) \
                    / np.linalg.norm(reference['x_force'])
        
        self.assertGreater(ft_err, 100*self.rel_force_tol,
                        'Tangent force should change for different asperities')
        
        fn_err = np.linalg.norm(fxyn_t[:, 2] 
                                - np.asarray(reference['normal_force'])) \
                    / reference['normal_force'] / np.sqrt(fxyn_t.shape[0])
        
        self.assertGreater(fn_err, 100*self.rel_force_tol,
                        'Normal force should change for different asperities')
        

    def test_decrease_normal_asperity(self):
        """
        Test a series of applied displacements for tangential MIF model
        at decreasing normal load.
        """
        
        pass

    def test_increase_normal_asperity(self):
        """
        Test a series of applied displacements for tangential MIF model
        at constant normal load.
        """
        
        pass
    
    def test_static_force(self):
        # Check gradients here
        
        pass
    
    def test_aft(self):
        # Check gradients here
        pass

if __name__ == '__main__':
    unittest.main()
