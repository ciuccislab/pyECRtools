import numpy as np
from numpy.linalg import inv, det, eig, cond
from .ecr_base import EcrBase  # Adjust the import path as needed

class EcrSens(EcrBase):
    """
    Python translation of the MATLAB class ecr_sens, extending EcrBase.
    It adds sensitivity-related methods such as gradient computation,
    covariance matrix determination, and noise boundary estimates.

    Notes:
        - 'cond_cov_sigma_n' uses numpy.linalg.cond to approximate MATLAB's condest.
        - 'max_eig_cov_sigma_n' uses the largest eigenvalue from np.linalg.eig.
        - 'boundary_cov_sigma_n' reconstructs the 2D ellipse boundary
          for (theta_k, theta_D) from the inverse covariance matrix.
    """

    def grad_sigma_n(self, time):
        """
        Compute the finite-difference gradient of sigma_n with respect to
        theta_k and theta_D.

        Args:
            time (array-like): Array of time points.

        Returns:
            np.ndarray of shape (len(time), 2): 
                Columns correspond to the partial derivatives wrt theta_k and theta_D.
        """
        time = np.asarray(time)
        delta = 1e-3

        # sigma_n(theta_k=1+delta, theta_D=1)
        sigma_n_plus_k = self.sigma_n_det_theta(np.array([1.0 + delta, 1.0]), time)
        # sigma_n(theta_k=1-delta, theta_D=1)
        sigma_n_minus_k = self.sigma_n_det_theta(np.array([1.0 - delta, 1.0]), time)

        # sigma_n(theta_k=1, theta_D=1+delta)
        sigma_n_plus_d = self.sigma_n_det_theta(np.array([1.0, 1.0 + delta]), time)
        # sigma_n(theta_k=1, theta_D=1-delta)
        sigma_n_minus_d = self.sigma_n_det_theta(np.array([1.0, 1.0 - delta]), time)

        # Finite-difference approximations
        grad_sigma_k = (sigma_n_plus_k - sigma_n_minus_k) / (2.0 * delta)
        grad_sigma_d = (sigma_n_plus_d - sigma_n_minus_d) / (2.0 * delta)

        return np.column_stack((grad_sigma_k, grad_sigma_d))

    def cov_sigma_n(self, time):
        """
        Estimate the covariance of (theta_k, theta_D) using a linearized approach.

        Args:
            time (array-like): Time points.

        Returns:
            2x2 covariance matrix for (theta_k, theta_D).
        """
        time = np.asarray(time)
        grad_mat = self.grad_sigma_n(time)

        # grad_sigma_n_theta_k and grad_sigma_n_theta_D
        grad_k = grad_mat[:, 0]
        grad_d = grad_mat[:, 1]

        # Form A_theta = [<dk,dk>, <dk,dd>; <dd,dk>, <dd,dd>]
        # Here <a,b> = sum(a[i]*b[i]) is the dot product over time points
        A_theta = np.array([[np.dot(grad_k, grad_k), np.dot(grad_k, grad_d)],
                            [np.dot(grad_d, grad_k), np.dot(grad_d, grad_d)]])

        # If the condition number is huge, revert to a fallback large identity
        if cond(A_theta) > 1e25:
            return 1e20 * np.eye(2)

        # Otherwise: Cov = (std_dev^2) * A_theta^-1
        return (self.standard_dev ** 2) * inv(A_theta)

    def det_cov_sigma_n(self, time):
        """
        Compute the determinant of the covariance matrix.

        Args:
            time (array-like): Time points.

        Returns:
            float: det of the covariance matrix.
        """
        matrix_V = self.cov_sigma_n(time)
        return det(matrix_V)

    def trace_cov_sigma_n(self, time):
        """
        Compute the trace of the covariance matrix.

        Args:
            time (array-like): Time points.

        Returns:
            float: Trace of the covariance matrix.
        """
        matrix_V = self.cov_sigma_n(time)
        return np.trace(matrix_V)

    def cond_cov_sigma_n(self, time):
        """
        Compute the condition number of the covariance matrix.

        Args:
            time (array-like): Time points.

        Returns:
            float: Condition number (2-norm) of the covariance matrix.
        """
        matrix_V = self.cov_sigma_n(time)
        return cond(matrix_V)

    def max_eig_cov_sigma_n(self, time):
        """
        Compute the largest eigenvalue of the covariance matrix.

        Args:
            time (array-like): Time points.

        Returns:
            float: Maximum eigenvalue of the covariance matrix.
        """
        matrix_V = self.cov_sigma_n(time)
        eigenvalues, _ = eig(matrix_V)
        return np.max(eigenvalues.real)

    def boundary_cov_sigma_n(self, time):
        """
        Construct a 2D ellipse boundary (3-sigma ellipse)
        for (theta_k, theta_D) from the inverse covariance matrix.

        Args:
            time (array-like): Time points.

        Returns:
            np.ndarray of shape (2, 101): The x and y coordinates of the boundary.
        """
        matrix_V = self.cov_sigma_n(time)
        inv_matrix_V = inv(matrix_V)

        # Eigen-decomposition of the inverse covariance
        vals, X_mat = eig(inv_matrix_V)

        # For a 2x2 system, vals and X_mat have length 2
        # 3-sigma ellipse in principal coordinates
        theta = np.linspace(0, 2.0 * np.pi, 101)
        x_prime = 3.0 / np.sqrt(vals[0]) * np.cos(theta)
        y_prime = 3.0 / np.sqrt(vals[1]) * np.sin(theta)

        # Transform back to (theta_k, theta_D) coordinates
        xy_prime = np.vstack((x_prime, y_prime))
        xy = X_mat @ xy_prime

        # Shift the center to (1.0, 1.0)
        xy[0, :] += 1.0
        xy[1, :] += 1.0

        return xy

    def sigma_n_det_theta(self, theta, time):
        """
        Compute sigma_n_det for custom (theta_k, theta_D) scaling factors
        relative to self.k_ref and self.D_ref.

        Args:
            theta (np.ndarray): [theta_k, theta_D].
            time (array-like): Array of time points.

        Returns:
            np.ndarray: Deterministic sigma_n for each time point.
        """
        time = np.asarray(time)

        half_x = self.half_thickness_x_ref
        half_y = self.half_thickness_y_ref
        half_z = self.half_thickness_z_ref

        theta_k, theta_D = theta[0], theta[1]

        # Check geometry
        if np.isinf(half_x) or np.isinf(half_y) or np.isinf(half_z):
            # 2D if exactly one dimension is finite, 1D if exactly one dimension is infinite, etc.
            # The code below matches the logic in the MATLAB version.
            if (half_x == np.inf and half_y < np.inf and half_z < np.inf) or \
               (half_y == np.inf and half_x < np.inf and half_z < np.inf) or \
               (half_z == np.inf and half_x < np.inf and half_y < np.inf):
                return self._sigma_n_det_2D_theta(theta_k, theta_D, time)
            else:
                return self._sigma_n_det_1D_theta(theta_k, theta_D, time)
        else:
            # 3D scenario: check for symmetrical or unsymmetrical geometry
            if (half_x == half_y) and (half_x == half_z):
                return self._sigma_n_det_symm_theta(theta_k, theta_D, time)
            else:
                return self._sigma_n_det_unsymm_theta(theta_k, theta_D, time)

    # -------------------------------------------------------------------------
    # Private-like methods for sigma_n with (theta_k, theta_D) scaling
    # -------------------------------------------------------------------------
    def _sigma_n_det_unsymm_theta(self, theta_k, theta_D, time):
        """
        sigma_n for a rectangular box with different side lengths
        and scaled parameters (k, D).
        """
        k = theta_k * self.k_ref
        D = theta_D * self.D_ref

        half_x = self.half_thickness_x_ref
        half_y = self.half_thickness_y_ref
        half_z = self.half_thickness_z_ref

        L_x = half_x * k / D
        L_y = half_y * k / D
        L_z = half_z * k / D

        beta_x = self.diffbeta(L_x)
        beta_y = self.diffbeta(L_y)
        beta_z = self.diffbeta(L_z)

        coeff_x = 2 * L_x**2 / (beta_x**2 * (beta_x**2 + L_x**2 + L_x))
        coeff_y = 2 * L_y**2 / (beta_y**2 * (beta_y**2 + L_y**2 + L_y))
        coeff_z = 2 * L_z**2 / (beta_z**2 * (beta_z**2 + L_z**2 + L_z))

        out_sigma_n = np.zeros_like(time, dtype=float)
        for i, t in enumerate(time):
            exp_x = coeff_x * np.exp(-(beta_x**2) * D * t / (half_x**2))
            exp_y = coeff_y * np.exp(-(beta_y**2) * D * t / (half_y**2))
            exp_z = coeff_z * np.exp(-(beta_z**2) * D * t / (half_z**2))
            out_sigma_n[i] = 1.0 - (np.sum(exp_x) * np.sum(exp_y) * np.sum(exp_z))
        return out_sigma_n

    def _sigma_n_det_symm_theta(self, theta_k, theta_D, time):
        """
        sigma_n for a cube with half_thickness_x = half_thickness_y = half_thickness_z
        and scaled parameters (k, D).
        """
        k = theta_k * self.k_ref
        D = theta_D * self.D_ref
        half_x = self.half_thickness_x_ref

        L_x = half_x * k / D
        beta_x = self.diffbeta(L_x)
        coeff_x = 2 * L_x**2 / (beta_x**2 * (beta_x**2 + L_x**2 + L_x))

        out_sigma_n = np.zeros_like(time, dtype=float)
        for i, t in enumerate(time):
            exp_x = coeff_x * np.exp(-(beta_x**2) * D * t / (half_x**2))
            # 3D symmetrical geometry => (1 - sum(exp_x))^3
            out_sigma_n[i] = 1.0 - (np.sum(exp_x))**3
        return out_sigma_n

    def _sigma_n_det_1D_theta(self, theta_k, theta_D, time):
        """
        sigma_n for a slab with effectively 1D transport,
        picking the smallest finite thickness among (x, y, z).
        """
        k = theta_k * self.k_ref
        D = theta_D * self.D_ref

        half_x = np.min([self.half_thickness_x_ref,
                         self.half_thickness_y_ref,
                         self.half_thickness_z_ref])

        L_x = half_x * k / D
        beta_x = self.diffbeta(L_x)
        coeff_x = 2 * L_x**2 / (beta_x**2 * (beta_x**2 + L_x**2 + L_x))

        out_sigma_n = np.zeros_like(time, dtype=float)
        for i, t in enumerate(time):
            exp_x = coeff_x * np.exp(-(beta_x**2) * D * t / (half_x**2))
            out_sigma_n[i] = 1.0 - np.sum(exp_x)
        return out_sigma_n

    def _sigma_n_det_2D_theta(self, theta_k, theta_D, time):
        """
        sigma_n for a slab with effectively 2D transport,
        choosing the two finite thicknesses among (x, y, z).
        """
        k = theta_k * self.k_ref
        D = theta_D * self.D_ref

        thicknesses = np.array([self.half_thickness_x_ref,
                                self.half_thickness_y_ref,
                                self.half_thickness_z_ref])
        finite_mask = np.isfinite(thicknesses)
        finite_vals = thicknesses[finite_mask]
        half_x = finite_vals[0]
        half_y = finite_vals[1]

        L_x = half_x * k / D
        L_y = half_y * k / D

        beta_x = self.diffbeta(L_x)
        beta_y = self.diffbeta(L_y)
        coeff_x = 2 * L_x**2 / (beta_x**2 * (beta_x**2 + L_x**2 + L_x))
        coeff_y = 2 * L_y**2 / (beta_y**2 * (beta_y**2 + L_y**2 + L_y))

        out_sigma_n = np.zeros_like(time, dtype=float)
        for i, t in enumerate(time):
            exp_x = coeff_x * np.exp(-(beta_x**2) * D * t / (half_x**2))
            exp_y = coeff_y * np.exp(-(beta_y**2) * D * t / (half_y**2))
            out_sigma_n[i] = 1.0 - (np.sum(exp_x) * np.sum(exp_y))
        return out_sigma_n