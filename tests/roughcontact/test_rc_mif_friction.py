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
import tmdsimpy.utils.harmonic as hutils

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
        self.rtol_grad = 1e-6
        
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
                                          gaps=np.array([0.0, 0.0, 0.0, 0.0]), 
                                          gap_weights=np.array([0.6, 0.25, 0.1, 0.05]),
                                          tangent_model='MIF',
                                          N_radial_quad=ref_dict['asp_num_quad_points'])
            
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
                                          tangent_model='MIF',
                                          N_radial_quad=ref_dict['asp_num_quad_points'])
        
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
                                          tangent_model='MIF',
                                          N_radial_quad=ref_dict['asp_num_quad_points'])
        
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
                                          tangent_model='MIF',
                                          N_radial_quad=ref_dict['asp_num_quad_points'])
            
        self.model_correct_asps = [correct_model1, 
                                   correct_model2, 
                                   correct_model3]
        
        self.model_diff_asps = diff_model
        
        self.ref_dict = ref_dict
        
        
        
        
        # Models for gradient checking, use fewer sliders to speed up
        # gradient checking time to keep reasonable
        N_radial_quad_fast = 10
        
        correct_model1_fast = RoughContactFriction(Q, T, 
                                ref_dict['E'], 
                                ref_dict['nu'], 
                                ref_dict['R'], 
                                ref_dict['Et'], 
                                np.float64(ref_dict['Sys']), 
                                mu=ref_dict['mu'], 
                                u0=0, 
                                meso_gap=0, 
                                gaps=np.array([0.0, 0.0, 0.0, 0.0]), 
                                gap_weights=np.array([0.6, 0.25, 0.1, 0.05]),
                                tangent_model='MIF',
                                N_radial_quad=N_radial_quad_fast)
            
        
        diff_model_fast = RoughContactFriction(Q, T, 
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
                                tangent_model='MIF',
                                N_radial_quad=N_radial_quad_fast)
        
        self.fast_models = [correct_model1_fast, diff_model_fast]
            


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
        
        unlth0 = np.array([0.0, 0.0, 0.0])
        
        reference = self.ref_dict['decreasing_normal']
        
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
                        / np.linalg.norm(reference['normal_force'])
            
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
                    / np.linalg.norm(reference['normal_force'])
        
        self.assertGreater(fn_err, 100*self.rel_force_tol,
                        'Normal force should change for different asperities')
        

    def test_increase_normal_asperity(self):
        """
        Test a series of applied displacements for tangential MIF model
        at constant normal load.
        """
        
        unlth0 = np.array([0.0, 0.0, 0.0])
        
        reference = self.ref_dict['increasing_normal']
        
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
                        / np.linalg.norm(reference['normal_force'])
            
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
                    / np.linalg.norm(reference['normal_force'])
        
        self.assertGreater(fn_err, 100*self.rel_force_tol,
                        'Normal force should change for different asperities')

    
    def test_static_force(self):
        """
        Check the static forces and that history is correctly saved and 
        restarted (compare to local force history)
        """
        
        unlth0 = np.array([0.0, 0.0, 0.0])
        
        test_key_list = ['constant_normal', 'increasing_normal']
        
        for key in test_key_list:
            
            reference = self.ref_dict[key]
            
            unlt = np.vstack((np.asarray(reference['x_disp']), 
                              0*np.asarray(reference['x_disp']),
                              np.asarray(reference['normal_disp']))).T
            
            model_list = [self.model_correct_asps[0], self.model_diff_asps]
            
            for ind, model in enumerate(model_list):
                
                fxyn_t = model.local_force_history(unlt, 0*unlt, 0, 0, unlth0, 
                                                   max_repeats=1)[0]
                
                model.init_history()
                
                fnlt_static = np.zeros_like(unlt)
                
                for time_ind in range(unlt.shape[0]):
                    
                    fnlt_static[time_ind, :] = model.force(unlt[time_ind, :], 
                                                           update_hist=True)[0]
                    
                ft_err = np.linalg.norm(fxyn_t - fnlt_static) \
                            / np.linalg.norm(fxyn_t)
                
                self.assertLess(ft_err, self.rel_force_tol,
                                'Static force is not consistent with local '
                                + 'force history for model: {}'.format(ind))
                
    
    def test_static_grad(self):
        """
        Check the static forces and that history is correctly saved and 
        restarted (compare to local force history)
        """
                
        test_key_list = ['constant_normal', 'increasing_normal']
        
        for key in test_key_list:
            
            reference = self.ref_dict[key]
            
            unlt = np.vstack((np.asarray(reference['x_disp']), 
                              -0.25*np.asarray(reference['x_disp']),
                              np.asarray(reference['normal_disp']))).T
            
            model_list = [self.model_correct_asps[0], self.model_diff_asps]
            
            num_failed_tight = 0
            
            for ind, model in enumerate(model_list):
                
                model.init_history()
                
                # Check static force gradient just at a subset of times
                for time_ind in range(0, unlt.shape[0], 10):
                    
                    fun = lambda X : model.force(X, update_hist=False)
                    
                    grad_failed = vutils.check_grad(fun, unlt[time_ind, :], 
                                                    verbose=False, 
                                                    atol=self.atol_grad,
                                                    rtol=0.0055, # self.rtol_grad,
                                                    h=1e-12)
                    
                    num_failed_tight += vutils.check_grad(fun, unlt[time_ind, :], 
                                                    verbose=False, 
                                                    atol=self.atol_grad,
                                                    rtol=self.rtol_grad,
                                                    h=1e-12,
                                                    silent=True)
                    
                    self.assertFalse(grad_failed, 
                                'Incorrect Gradient w.r.t. U for AFT, '
                                + 'model: {}, time: {}'.format(ind, time_ind))
                    
                    model.force(unlt[time_ind, :], update_hist=True)
            
            self.assertLessEqual(num_failed_tight, 9, 
                                 'More than the expected number of gradient '
                                 + 'checks required looser tolerances')


    def test_aft_versus_local_hist(self):
        """
        Check that AFT and local force history give consistent results
        """
        w = 1.35
        h = np.array(range(5+1))
        Nt = 1 << 7
        
        Ndof = 3
        Nhc = hutils.Nhc(h)

        U_subset = np.array([0, .5e-5, -.1e-5, 
                            0.1e-5, 0.2e-5, -0.1e-5, 
                            0.3e-5, 0.2e-5, -0.1e-5, 
                            0.1e-5, 0.1e-5, -0.5e-5])
        
        U = np.zeros(Ndof * Nhc)
        U[:U_subset.shape[0]] = U_subset
        
        # Time series from harmonic displacements
        Ulocal = np.reshape(U, (Ndof, Nhc), 'F').T
        unlt = hutils.time_series_deriv(Nt, h, Ulocal, 0) # Nt x Ndnl
        
        unlth0 = np.zeros(3)
        
        model_list = [self.model_correct_asps[0]] + [self.model_diff_asps]

        for ind, model in enumerate(model_list):
            
            fxyn_t = model.local_force_history(unlt, 0, h, 0, unlth0)[0]
            
            
            Fnl_local = hutils.get_fourier_coeff(h, fxyn_t)
            Fnl_local = np.reshape(Fnl_local.T, (-1,), 'F')
            
            Fnl_aft = model.aft(U, w, h)[0]
            
            self.assertLess(np.linalg.norm(Fnl_local - Fnl_aft), 1e-10,
                        'AFT is not consistent with the local force history.')
    
    def test_aft_grad(self):
        """
        Test for AFT Gradients with Mindlin Iwan Fit asperity tangential 
        contact model
        """
        
        # Check gradients here
        model_list = self.fast_models
        
        U_list = [np.array([0, .5e-5, 2e-5, 
                          0.1e-5, 0.2e-5, 0.05e-5, 
                          0.3e-5, 0.2e-5, 0.05e-5, 
                          0.0, 0.0, 0.0, 
                          0.0, 0.0, 0.0,
                          0.1e-5, 0.1e-5, 0.02e-5]),
                  np.array([0, .5e-5, -.1e-5, 
                          0.1e-5, 0.2e-5, -0.1e-5, 
                          0.3e-5, 0.2e-5, -0.1e-5, 
                          0.1e-5, 0.1e-5, -0.5e-5]),
                  np.array([0, .5e-5, 2e-5, 
                          0.0e-5, 0.0e-5, -0.0e-5, 
                          0.4e-5, 0.4e-5, 0.08e-5, 
                          0.0e-5, 0.0e-5, -0.0e-5,
                          0.1e-5, 0.1e-5, 0.02e-5]),
                  np.array([0, .5e-5, 2e-5, 
                          0.0e-5, 0.0e-5, 0.0e-5, 
                          0.4e-5, 0.4e-5, 0.08e-5, 
                          0.0e-5, 0.0e-5, 0.0e-5,
                          0.0e-5, 0.0e-5, 0.0e-5,
                          0.0e-5, 0.0e-5, 0.0e-5,
                          0.1e-5, 0.1e-5, 0.02e-5]),
                ]
        
        w = 1.35
        h = np.array(range(5+1))
        Nt = 1 << 7
        
        Ndof = 3
        Nhc = hutils.Nhc(h)

        # import pdb; pdb.set_trace()

        for ind, model in enumerate(model_list):
            for j in range(len(U_list)):
                
                
                U = np.zeros(Ndof * Nhc)
                U[:U_list[j].shape[0]] = U_list[j]
                
                fun = lambda U : model.aft(U, w, h, Nt=Nt, max_repeats=2)[0:2]
                
                loosen_grad = 1
                # if j == 6:
                #     loosen_grad = 1e3
                
                grad_failed = vutils.check_grad(fun, U, verbose=False, 
                                            atol=self.atol_grad,
                                            rtol=self.rtol_grad*loosen_grad,
                                            h=1e-6*np.linalg.norm(U))
                
                self.assertFalse(grad_failed, 
                                 'Incorrect Gradient w.r.t. U for AFT, '
                                 + 'model: {}, set: {}'.format(ind, j))
                
                dFdw = model.aft(U, w, h, Nt=Nt, max_repeats=2)[2]
                
                self.assertLess(np.linalg.norm(dFdw), 1e-12, 
                                'Gradient w.r.t. w should be zero.'
                                + ' Model: {}, set: {}'.format(ind, j))

if __name__ == '__main__':
    unittest.main()
