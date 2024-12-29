'''
demo 3:
1. compute asymptotic covariance matrix metrics: det(V), trace(V), cond(V), max_eig(V)
2. vary the sample half-thickness from 1e-4 m to 1e-2 m
3. plot these metrics vs. the half-thickness
'''

import numpy as np
import matplotlib.pyplot as plt

from ECR_classes import EcrSens

def demo_3():
    # 1) instantiate ecr_sens
    a = EcrSens()

    # 2) parameters
    a.k_ref = 1.5e-4
    a.D_ref = 2.5e-9
    a.standard_dev = 1e-2

    # 3) time array
    t_min = 0.0
    t_max = 1e4
    # in matlab, the code uses 1e4+1 => 10001 points
    time = np.linspace(t_min, t_max, int(1e4 + 1))

    # 4) log-spaced half-thickness vector
    half_thickness_min = 1e-4
    half_thickness_max = 1e-2
    N_ht = 11
    half_thickness_vec = np.logspace(np.log10(half_thickness_min),
                                     np.log10(half_thickness_max),
                                     N_ht)

    # prepare arrays
    det_cov_sigma_n_vec = np.zeros(N_ht)
    trace_cov_sigma_n_vec = np.zeros(N_ht)
    cond_cov_sigma_n_vec = np.zeros(N_ht)
    max_eig_cov_sigma_n_vec = np.zeros(N_ht)

    # 5) loop over half_thickness values
    for i, h in enumerate(half_thickness_vec):
        # set geometry to 3d cube
        a.half_thickness_x_ref = h
        a.half_thickness_y_ref = h
        a.half_thickness_z_ref = h

        # compute and store
        det_cov_sigma_n_vec[i]   = a.det_cov_sigma_n(time)
        trace_cov_sigma_n_vec[i] = a.trace_cov_sigma_n(time)
        cond_cov_sigma_n_vec[i]  = a.cond_cov_sigma_n(time)
        max_eig_cov_sigma_n_vec[i] = a.max_eig_cov_sigma_n(time)

        print(f'iteration {i+1}/{N_ht} => half_thickness={h:g}')

    # 6) plot
    # font setup for plots
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=20)
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # a) det(V)
    axs[0, 0].loglog(half_thickness_vec, det_cov_sigma_n_vec, '-k', linewidth=2)
    axs[0, 0].set_xlim([1e-4, 1e-2])
    axs[0, 0].set_xlabel('thickness (m)')
    axs[0, 0].set_ylabel('det(V)')

    # b) trace(V)
    axs[0, 1].loglog(half_thickness_vec, trace_cov_sigma_n_vec, '-k', linewidth=2)
    axs[0, 1].set_xlim([1e-4, 1e-2])
    axs[0, 1].set_ylim([1e-3, 1e0])
    axs[0, 1].set_xlabel('thickness (m)')
    axs[0, 1].set_ylabel('trace(V)')

    # c) cond(V)
    axs[1, 0].loglog(half_thickness_vec, cond_cov_sigma_n_vec, '-k', linewidth=2)
    axs[1, 0].set_xlim([1e-4, 1e-2])
    axs[1, 0].set_ylim([1e-1, 1e5])
    axs[1, 0].set_xlabel('thickness (m)')
    axs[1, 0].set_ylabel(r'$\kappa(V)$')

    # d) max eigenvalue
    axs[1, 1].loglog(half_thickness_vec, max_eig_cov_sigma_n_vec, '-k', linewidth=2)
    axs[1, 1].set_xlim([1e-4, 1e-2])
    axs[1, 1].set_ylim([1e-3, 1e0])
    axs[1, 1].set_xlabel('thickness (m)')
    axs[1, 1].set_ylabel(r'$\max(\lambda(V))$')

    plt.suptitle('demo 3: covariance metrics vs. half-thickness')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    demo_3()