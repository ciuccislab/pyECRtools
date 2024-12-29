'''
demo 4:
1. read in measurement data from file (data.txt)
2. fit it using log10_fit and then a more precise fit
3. update k_ref and d_ref based on the best fit
4. plot the data, fit, and residual
5. plot the 2d confidence ellipse (coVariance boundary)
'''

import numpy as np
import matplotlib.pyplot as plt

from ECR_classes import EcrFit

def demo_4():
    # 1) load data from text file
    # the matlab code does: ecr_data = load('data.txt')
    # => columns: time_exp, sigma_n_exp
    # we'll replicate that in python:
    ECR_data = np.loadtxt('data.txt', delimiter=None)  # or you might need a specific delimiter
    # shift time so that 0 is the first measurement time
    time_exp = ECR_data[:,0] - ECR_data[0,0]
    sigma_n_exp = ECR_data[:,1]

    # 2) instantiate ecr_fit
    a = EcrFit()
    # set initial references
    a.k_ref = 1e-4   # [m/s]
    a.D_ref = 1e-6   # [m^2/s]
    a.half_thickness_x_ref = 1e-3
    a.half_thickness_y_ref = 1e-3
    a.half_thickness_z_ref = 1e-3

    # preliminary guess of std
    a.standard_deV = 1e-2

    # attach measurements
    a.meas_time = time_exp
    a.meas_sigma_n = sigma_n_exp

    # font setup for plots
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=20)
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)

    # 3) first pass: log10 fit to get in the right ballpark
    theta_out = a.log10_fit(method='TNC')
    # update references
    a.k_ref *= theta_out[0]
    a.D_ref *= theta_out[1]

    # 4) more accurate fit
    theta_out_lin = a.fit(method='TNC')
    # update references again
    a.k_ref *= theta_out_lin[0]
    a.D_ref *= theta_out_lin[1]

    # eValuate final model
    sigma_computed = a.output_sigma_n_det(theta_out_lin)
    error_meas = sigma_computed - sigma_n_exp

    # recompute standard deViation from residual
    std_computed = a.compute_std_meas()
    a.standard_deV = std_computed

    # 5) plot data Vs. fit, plus residual on secondary axis
    fig, ax1 = plt.subplots(figsize=(8,6))
    color1 = 'tab:blue'
    color2 = 'tab:red'

    ax1.set_xlabel(r'$t_{\rm exp}$ (s)')
    ax1.set_ylabel(r'$\sigma_n$', color=color1)
    l1 = ax1.plot(time_exp, sigma_computed, '-k', linewidth=3, label='fit model')
    ax1.plot(time_exp, sigma_n_exp, '+r', markersize=5, label='measured data')
    ax1.tick_params(axis='y', labelcolor=color1)

    # set x-axis range and ticks
    ax1.set_xlim([0, time_exp.max()])  # set x-axis range
    ax1.set_xticks(np.arange(0, time_exp.max() + 1, 200))  # set x-axis ticks with step 200

    # set y-axis range and ticks for ax1
    ax1.set_ylim([0, 1])  # set y-axis range
    ax1.set_yticks(np.arange(0, 1.1, 0.1))  # set y-axis ticks from 0 to 1 with step 0.1

    # twin axis for residual
    ax2 = ax1.twinx()
    ax2.set_ylabel(r'$100\times(\sigma_n - \sigma_n^{\rm meas})$', color=color2)
    l2 = ax2.plot(time_exp, 100.0 * error_meas, '+b', markersize=4, label=r'residual $\times$ 100')
    ax2.tick_params(axis='y', labelcolor=color2)

    # set y-axis range and range for ax2
    ax2.set_ylim([-100, 100])  # set residual y-axis range from -100 to 100
    ax2.set_yticks(np.arange(-100, 110, 10))  # y-axis ticks from -100 to 100 with step 10

    # add some lines for ±3 std (in the residual space)
    ax2.axhline(100.0 * (+3.0 * std_computed), color='g', linestyle='--', linewidth=2)
    ax2.axhline(100.0 * (-3.0 * std_computed), color='g', linestyle='--', linewidth=2)

    # combine legend
    lines = l1 + l2
    labs  = [line.get_label() for line in lines]
    ax1.legend(lines, labs, loc='best')
    plt.title('demo 4: fit data and residual')

    plt.tight_layout()
    plt.show()

    # 6) confidence ellipse
    boundary_cov = a.boundary_cov_sigma_n(time_exp)

    plt.figure(figsize=(6,5))
    plt.plot(boundary_cov[0,:], boundary_cov[1,:], '-k', linewidth=2)
    plt.xlabel(r'$\hat{k}/k_{\rm fit}$')
    plt.ylabel(r'$\hat{D}/D_{\rm fit}$', fontsize=14)
    plt.title('confidence ellipse (3-sigma)', fontsize=12)
    plt.axis([0,10, 0,10])
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    demo_4()