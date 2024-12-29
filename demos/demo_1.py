import numpy as np
import matplotlib.pyplot as plt
from ECR_classes import EcrBase

def demo_1():
    a = EcrBase()
    a.k_ref = 1.5e-4
    a.D_ref = 2.5e-9
    a.half_thickness_x_ref = 1e-3
    a.half_thickness_y_ref = 1e-3
    a.half_thickness_z_ref = 1e-3
    a.standard_dev = 1e-2

    N = 101
    time = np.linspace(0, 100, N)

    sigma_n = a.sigma_n_det(time)
    sigma_n_meas = a.sigma_n_meas(time)

    # font setup for plots
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=20)
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)

    plt.figure(figsize=(10, 10/1.618))
    plt.plot(time, sigma_n, '-k', linewidth=3, label='Exact $\\sigma_n$')
    plt.plot(time, sigma_n_meas, '.r', markersize=12, label='Measured $\\sigma_n$')
    plt.xlabel('$t$ (s)')
    plt.ylabel('$\\sigma_n$')
    plt.title('demo 1: normalized conductivity and synthetic measurements', fontsize=24)

    # set x and y axis limits and ticks
    plt.xlim(0, 100)
    plt.xticks(np.arange(0, 101, 10))
    plt.ylim(0.0, 1.0)
    plt.yticks(np.arange(0.0, 1.1, 0.2))
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    demo_1()