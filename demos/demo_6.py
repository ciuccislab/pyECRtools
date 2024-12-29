'''
demo 6:
1. perform synthetic experiments.
2. fit each synthetic experiment.
3. plot the outcome of these synthetic fits and compare them against
   the 3-sigma confidence boundary of the asymptotic covariance matrix.

requires:
 - ecr_fit (and its superclass ecr_sens, ecr_base)
 - numpy and matplotlib
'''

import numpy as np
import matplotlib.pyplot as plt
import time

from ECR_classes import EcrFit

def demo_6():
    # define experimental time
    t_exp = np.linspace(0, 500, 151)  # 151 points from 0 to 500 s

    # instantiate the fitting class
    a = EcrFit()
    a.k_ref = 1e-4
    a.D_ref = 1e-9  # Corrected to match demo_6_old.py
    a.half_thickness_x_ref = 1e-3
    a.half_thickness_y_ref = 1e-3
    a.half_thickness_z_ref = 1e-3
    a.standard_dev = 1e-2  # 1% noise
    a.meas_time = t_exp    # assign the time vector

    # obtain the 3-sigma covariance boundary at these times
    boundary_cov = a.boundary_cov_sigma_n(a.meas_time)
    # boundary_cov should be a 2 x n array: [ [theta_k_1, ..., theta_k_n],
    #                                        [theta_D_1,  ..., theta_D_n] ]

    # perform synthetic experiments
    print('running synthetic experiments...')
    start_time = time.time()
    theta_vec = a.synthetic_exp(N_exp=100)
    end_time = time.time()
    print(f'synthetic experiments completed in {end_time - start_time:.2f} s')

    # font setup for plots
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=20)
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    
    # plot the results
    plt.figure(figsize=(9,7))
    # plot covariance ellipse
    plt.plot(boundary_cov[0, :], boundary_cov[1, :],
             '-k', linewidth=2, label='3-sigma boundary')
    # plot recovered parameters
    plt.plot(theta_vec[0, :], theta_vec[1, :],
             '.r', markersize=8, label='Fitted synthetic data')

    plt.xlabel(r'$\hat{k}/k_{\mathrm{exact}}$', fontsize=16)
    plt.ylabel(r'$\hat{D}/D_{\mathrm{exact}}$', fontsize=16)
    plt.title('demo 6: synthetic experiments and covariance boundary', fontsize=14)
    plt.axis([0.4, 1.6, 0.4, 1.6])
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    demo_6()