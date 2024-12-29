'''
demo 5:
1. test multiple solvers for fitting experimental data.
2. output the fit of sigma and residual for each solver.
'''

import numpy as np
import matplotlib.pyplot as plt

from ECR_classes import EcrFit

def demo_5():
    # load data from text file
    ecr_data = np.loadtxt('data.txt', delimiter=None)
    time_exp = ecr_data[:, 0] - ecr_data[0, 0]  # shift time so that 0 is the first measurement time
    sigma_n_exp = ecr_data[:, 1]

    # instantiate ecr_fit
    a = EcrFit()
    a.k_ref = 1e-4  # [m/s]
    a.d_ref = 1e-6  # [m^2/s]
    a.half_thickness_x_ref = 1e-3
    a.half_thickness_y_ref = 1e-3
    a.half_thickness_z_ref = 1e-3
    a.standard_dev = 1e-2  # preliminary guess of std

    a.meas_time = time_exp
    a.meas_sigma_n = sigma_n_exp

    # font setup for plots
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=20)
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)

    # test multiple solvers for fitting
    solvers = [
        'Nelder-Mead', 'Powell', 'L-BFGS-B', 'TNC',
        'COBYLA', 'SLSQP', 'trust-constr'
    ]

    for method in solvers:
        print(f"testing solver: {method}")
        # first pass: log10 fit
        theta_out = a.log10_fit(method=method)
        a.k_ref *= theta_out[0]
        a.d_ref *= theta_out[1]

        # more accurate fit
        theta_out_lin = a.fit(method=method)
        a.k_ref *= theta_out_lin[0]
        a.d_ref *= theta_out_lin[1]

        # evaluate final model
        sigma_computed = a.output_sigma_n_det(theta_out_lin)
        error_meas = sigma_computed - sigma_n_exp

        # recompute standard deviation from residual
        std_computed = a.compute_std_meas()
        a.standard_dev = std_computed

        # plot data vs. fit and residual
        fig, ax1 = plt.subplots(figsize=(8, 6))
        color1 = 'tab:blue'
        color2 = 'tab:red'

        ax1.set_xlabel(r'$t_{\rm exp}$ (s)')
        ax1.set_ylabel(r'$\sigma_n$', color=color1)
        l1 = ax1.plot(time_exp, sigma_computed, '-k', linewidth=3, label='fit model')
        ax1.plot(time_exp, sigma_n_exp, '+r', markersize=5, label='measured data')
        ax1.tick_params(axis='y', labelcolor=color1)

        # set x-axis range and ticks
        ax1.set_xlim([0, time_exp.max()])
        ax1.set_xticks(np.arange(0, time_exp.max() + 1, 200))

        # set y-axis range and ticks for ax1
        ax1.set_ylim([0, 1])
        ax1.set_yticks(np.arange(0, 1.1, 0.1))

        # twin axis for residual
        ax2 = ax1.twinx()
        ax2.set_ylabel(r'$100\times(\sigma_n - \sigma_n^{\rm meas})$', color=color2)
        l2 = ax2.plot(time_exp, 100.0 * error_meas, '+b', markersize=4, label=r'residual $\times$ 100')
        ax2.tick_params(axis='y', labelcolor=color2)

        # set y-axis range and range for ax2
        ax2.set_ylim([-100, 100])
        ax2.set_yticks(np.arange(-100, 110, 10))

        # add lines for ±3 std in residual space
        ax2.axhline(100.0 * (+3.0 * std_computed), color='g', linestyle='--', linewidth=2)
        ax2.axhline(100.0 * (-3.0 * std_computed), color='g', linestyle='--', linewidth=2)

        # combine legend
        lines = l1 + l2
        labs = [line.get_label() for line in lines]
        ax1.legend(lines, labs, loc='best')

        # set title and save figure
        plt.title(f'demo 5: fit data and residual (method: {method})')
        plt.tight_layout()
        import os
        output_dir = os.path.join('demos', 'figs_demo')
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"demo_5_{method}.png"))
        plt.close()

if __name__ == "__main__":
    demo_5()
