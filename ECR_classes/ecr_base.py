import numpy as np
from scipy.optimize import root_scalar

class EcrBase:
    def __init__(self, k_ref=None, D_ref=None, 
                 half_thickness_x_ref=None, 
                 half_thickness_y_ref=None, 
                 half_thickness_z_ref=None, 
                 standard_dev=None):
        self.k_ref = k_ref if k_ref is not None else 1.0
        self.D_ref = D_ref if D_ref is not None else 1.0
        self.half_thickness_x_ref = half_thickness_x_ref if half_thickness_x_ref is not None else 1.0
        self.half_thickness_y_ref = half_thickness_y_ref if half_thickness_y_ref is not None else 1.0
        self.half_thickness_z_ref = half_thickness_z_ref if half_thickness_z_ref is not None else 1.0
        self.standard_dev = standard_dev if standard_dev is not None else 0.01

    def sigma_n_det(self, time):
        # Convert time to numpy array if it isn't already
        time = np.asarray(time)
        
        half_thickness_x = self.half_thickness_x_ref
        half_thickness_y = self.half_thickness_y_ref
        half_thickness_z = self.half_thickness_z_ref

        if np.isinf(half_thickness_x) or np.isinf(half_thickness_y) or np.isinf(half_thickness_z):
            if np.isinf(half_thickness_x) and half_thickness_y < np.inf and half_thickness_z < np.inf:
                return self.sigma_n_det_2D(time)
            elif np.isinf(half_thickness_y) and half_thickness_x < np.inf and half_thickness_z < np.inf:
                return self.sigma_n_det_2D(time)
            elif np.isinf(half_thickness_z) and half_thickness_x < np.inf and half_thickness_y < np.inf:
                return self.sigma_n_det_2D(time)
            else:
                return self.sigma_n_det_1D(time)
        elif half_thickness_x == half_thickness_y == half_thickness_z:
            return self.sigma_n_det_symm(time)
        else:
            return self.sigma_n_det_unsymm(time)

    def sigma_n_meas(self, time):
        sigma_n_det = self.sigma_n_det(time)
        error_sigma = self.error_fct(sigma_n_det)
        return sigma_n_det + error_sigma

    def sigma_n_det_symm(self, time):
        k = self.k_ref
        D = self.D_ref
        L_x = self.half_thickness_x_ref * k / D
        beta_x = self.diffbeta(L_x)
        coeff_x = 2 * L_x**2 / (beta_x**2 * (beta_x**2 + L_x**2 + L_x))
        sigma_n = np.zeros_like(time)

        for i, t in enumerate(time):
            a_x = coeff_x * np.exp(-(beta_x**2 * D * t / self.half_thickness_x_ref**2))
            sigma_n[i] = 1 - np.sum(a_x)**3

        return sigma_n

    def sigma_n_det_unsymm(self, time):
        k = self.k_ref
        D = self.D_ref
        L_x = self.half_thickness_x_ref * k / D
        L_y = self.half_thickness_y_ref * k / D
        L_z = self.half_thickness_z_ref * k / D

        beta_x = self.diffbeta(L_x)
        beta_y = self.diffbeta(L_y)
        beta_z = self.diffbeta(L_z)

        coeff_x = 2 * L_x**2 / (beta_x**2 * (beta_x**2 + L_x**2 + L_x))
        coeff_y = 2 * L_y**2 / (beta_y**2 * (beta_y**2 + L_y**2 + L_y))
        coeff_z = 2 * L_z**2 / (beta_z**2 * (beta_z**2 + L_z**2 + L_z))

        sigma_n = np.zeros_like(time)
        for i, t in enumerate(time):
            a_x = coeff_x * np.exp(-(beta_x**2 * D * t / self.half_thickness_x_ref**2))
            a_y = coeff_y * np.exp(-(beta_y**2 * D * t / self.half_thickness_y_ref**2))
            a_z = coeff_z * np.exp(-(beta_z**2 * D * t / self.half_thickness_z_ref**2))
            sigma_n[i] = 1 - np.sum(a_x) * np.sum(a_y) * np.sum(a_z)

        return sigma_n

    def sigma_n_det_1D(self, time):
        k = self.k_ref
        D = self.D_ref
        half_thickness = min(x for x in [self.half_thickness_x_ref, 
                                       self.half_thickness_y_ref, 
                                       self.half_thickness_z_ref] if not np.isinf(x))
        L_x = half_thickness * k / D
        beta_x = self.diffbeta(L_x)
        coeff_x = 2 * L_x**2 / (beta_x**2 * (beta_x**2 + L_x**2 + L_x))
        sigma_n = np.zeros_like(time)

        for i, t in enumerate(time):
            a_x = coeff_x * np.exp(-(beta_x**2 * D * t / half_thickness**2))
            sigma_n[i] = 1 - np.sum(a_x)

        return sigma_n

    def sigma_n_det_2D(self, time):
        k = self.k_ref
        D = self.D_ref
        # Get the two finite thicknesses
        finite_thicknesses = [x for x in [self.half_thickness_x_ref, 
                                        self.half_thickness_y_ref, 
                                        self.half_thickness_z_ref] if not np.isinf(x)]
        L_x = finite_thicknesses[0] * k / D
        L_y = finite_thicknesses[1] * k / D

        beta_x = self.diffbeta(L_x)
        beta_y = self.diffbeta(L_y)

        coeff_x = 2 * L_x**2 / (beta_x**2 * (beta_x**2 + L_x**2 + L_x))
        coeff_y = 2 * L_y**2 / (beta_y**2 * (beta_y**2 + L_y**2 + L_y))

        sigma_n = np.zeros_like(time)
        for i, t in enumerate(time):
            a_x = coeff_x * np.exp(-(beta_x**2 * D * t / finite_thicknesses[0]**2))
            a_y = coeff_y * np.exp(-(beta_y**2 * D * t / finite_thicknesses[1]**2))
            sigma_n[i] = 1 - np.sum(a_x) * np.sum(a_y)

        return sigma_n

    def error_fct(self, sigma_n_det, integer_seed=42):
        np.random.seed(integer_seed) # For reproducibility
        return self.standard_dev * np.random.randn(*np.asarray(sigma_n_det).shape)

    @staticmethod
    def diffbeta(L):
        M = 80
        beta = np.zeros(M)

        if L < 1e-10:
            beta[0] = root_scalar(lambda x: x * np.tan(x) - L, bracket=[0, np.pi/2 - 1e-5]).root
            beta[1:] = np.arange(1, M) * np.pi
        elif L > 1e13:
            beta = (2 * np.arange(M) + 1) * np.pi / 2
        else:
            for n in range(M):
                try:
                    beta[n] = root_scalar(lambda x: x * np.tan(x) - L, 
                                        bracket=[n * np.pi, (n + 0.5) * np.pi]).root
                except:
                    beta[n] = n * np.pi

        return beta