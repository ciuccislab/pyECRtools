"""
demo 2:
1. use the ecr_sens class to compute sensitivity of sigma_n wrt. k and d
2. plot the sensitivities as a function of time (log scale)
"""

import numpy as np
import matplotlib.pyplot as plt

# from ecr_sens import ecrsens
from ECR_classes import EcrSens

def demo_2():
    # 1) instantiate the sensitivity class
    a = EcrSens()

    # 2) set parameters
    a.k_ref = 1.5e-4
    a.D_ref = 2.5e-9
    a.half_thickness_x_ref = 1e-3
    a.half_thickness_y_ref = 1e-3
    a.half_thickness_z_ref = 1e-3

    # 3) define timespan (logarithmically spaced)
    N = 1000
    t_min = 1e-2
    t_max = 1e4
    time = np.logspace(np.log10(t_min), np.log10(t_max), N)

    # 4) compute gradient wrt k and d
    grad_out = a.grad_sigma_n(time)
    # grad_out[:,0] -> d sigma_n / d theta_k
    # grad_out[:,1] -> d sigma_n / d theta_d

    # 5) plot results
    # font setup for plots
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=20)
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)

    plt.figure(figsize=(8,6))
    plt.semilogx(time, grad_out[:,0], '-k', linewidth=2, label=r'$q=k$')
    plt.semilogx(time, grad_out[:,1], '-r', linewidth=2, label=r'$q=D$')
    plt.xlabel(r'$t$ (s)')
    plt.ylabel(r'$q\frac{\partial \sigma_n}{\partial q}$')
    plt.xlim(1E-2, 1E4)
    plt.ylim(0.0, 0.25)
    plt.yticks(np.arange(0.0, 0.26, 0.05))
    plt.title(r'demo 2: sensitivity of $\sigma_n$ wrt $k$ and $d$', fontsize=24)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    demo_2()