import time
import numpy as np

# Assuming ecr_sens.py has:
#   class EcrSens(EcrBase):
#       ...
#       (Contains grad_sigma_n, cov_sigma_n, sigma_n_det, sigma_n_det_theta, etc.)
from .ecr_sens import EcrSens

class EcrFit(EcrSens):
    '''
    Python translation of the MATLAB class ecr_fit, which extends EcrSens.
    It adds methods for fitting experimental data and computing synthetic experiments.

    Attributes:
        meas_time (numpy.ndarray): The measurement time points.
        meas_sigma_n (numpy.ndarray): The measured sigma values at meas_time.
    '''
    def __init__(self,
                 k_ref=1.0,
                 D_ref=1.0,
                 half_thickness_x_ref=np.inf,
                 half_thickness_y_ref=np.inf,
                 half_thickness_z_ref=np.inf,
                 standard_dev=0.0,
                 meas_time=None,
                 meas_sigma_n=None):
        '''
        Initialize the EcrFit object.

        Args:
            k_ref (float): Reference value for k.
            D_ref (float): Reference value for D.
            half_thickness_x_ref (float): Reference half-thickness in x direction.
            half_thickness_y_ref (float): Reference half-thickness in y direction.
            half_thickness_z_ref (float): Reference half-thickness in z direction.
            standard_dev (float): Standard deviation for measurement noise.
            meas_time (array-like): Measurement time points.
            meas_sigma_n (array-like): Corresponding measured sigma_n values.
        '''
        super().__init__(k_ref=k_ref,
                         D_ref=D_ref,
                         half_thickness_x_ref=half_thickness_x_ref,
                         half_thickness_y_ref=half_thickness_y_ref,
                         half_thickness_z_ref=half_thickness_z_ref,
                         standard_dev=standard_dev)

        self.meas_time = np.asarray(meas_time) if meas_time is not None else None
        self.meas_sigma_n = np.asarray(meas_sigma_n) if meas_sigma_n is not None else None

    # -------------------------------------------------------------------------
    # Public Methods
    # -------------------------------------------------------------------------
    def log10_fit(self, method='TNC'):
        '''
        Fit (theta_k, theta_D) in log10 space by minimizing the distance
        between sigma_n(theta) and measured sigma_n. Replicates the
        dist_sigma_n_sigma_n_meas_log-based approach in MATLAB.

        Returns:
            numpy.ndarray: 1D array of fitted [theta_k, theta_D].
        '''
        def objective(log10_theta):
            return self._dist_sigma_n_sigma_n_meas_log(log10_theta)

        # Lower and upper bounds for log10(theta_k), log10(theta_D)
        lb_log10 = np.array([-7.0, -7.0])
        ub_log10 = np.array([ 7.0,  7.0])

        # Initial guess
        log10_theta_0 = np.array([0.0, 0.0])

        print('Pre-fitting data using (placeholder) method in log10 space...')

        # ---------------------------------------------------------
        # Example: Using scipy.optimize.minimize as a placeholder
        # You can switch to other libraries such as nlopt, pygmo, etc.
        # ---------------------------------------------------------
        from scipy.optimize import Bounds, minimize

        bounds = Bounds(lb_log10, ub_log10)
        start_time = time.process_time()

        res = minimize(
            fun=objective,
            x0=log10_theta_0,
            method=method,     # for example, 'TNC' or 'L-BFGS-B'
            bounds=bounds
        )

        elapsed_time = time.process_time() - start_time
        print(f'(time = {elapsed_time:.4f} s)')

        log10_theta_out = res.x
        theta_out = 10.0**log10_theta_out
        return theta_out

    def fit(self, method='TNC'):
        '''
        Fit (theta_k, theta_D) in linear space by minimizing the distance
        between sigma_n(theta) and measured sigma_n.

        Args:
            solver_used (str): String to specify which solver or method to use
                               (placeholder here for your chosen Python solver).

        Returns:
            numpy.ndarray: 1D array of fitted [theta_k, theta_D].
        '''
        def objective(theta):
            return self._dist_sigma_n_sigma_n_meas(theta)

        lb_theta = np.array([1e-3, 1e-3])
        ub_theta = np.array([1e3, 1e3])

        print(f'Fitting data using scipy with the ' + method + ' method')

        # Initial guess
        theta_0 = np.array([1.0, 1.0])

        # ---------------------------------------------------------
        # Example: Using scipy.optimize.minimize as a placeholder
        # (Replace or adapt as needed for other methods.)
        # ---------------------------------------------------------
        from scipy.optimize import Bounds, minimize
        bounds = Bounds(lb_theta, ub_theta)

        start_time = time.process_time()

        res = minimize(
            fun=objective,
            x0=theta_0,
            method='TNC',  # e.g., 'TNC', 'L-BFGS-B', or other
            bounds=bounds
        )
        elapsed_time = time.process_time() - start_time
        print(f'(time = {elapsed_time:.4f} s)')

        return res.x

    def synthetic_exp(self, method='TNC', N_exp=1, seed=42):
        if self.meas_time is None:
            raise ValueError('meas_time is not set.')

        time_exp = self.meas_time
        base_sigma_n_det = self.sigma_n_det(time_exp)
        theta_vec = np.zeros((2, N_exp))

        np.random.seed(seed)
        for i in range(N_exp):
            current_seed = np.random.randint(0, 1000000)
            noise = self.error_fct(base_sigma_n_det, integer_seed=current_seed)
            self.meas_sigma_n = base_sigma_n_det + noise

            print(f'Synthetic measurement {i+1}/{N_exp}')

            def objective(theta):
                return self._dist_sigma_n_sigma_n_meas(theta)

            lb_theta = np.array([1e-1, 1e-1])
            ub_theta = np.array([1e1, 1e1])
            theta_0 = np.array([1.0, 1.0])

            from scipy.optimize import Bounds, minimize
            bounds = Bounds(lb_theta, ub_theta)

            start_time = time.process_time()
            res = minimize(
                fun=objective,
                x0=theta_0,
                method=method,
                bounds=bounds
            )
            elapsed_time = time.process_time() - start_time
            print(f'(time = {elapsed_time:.4f} s)')

            theta_vec[:, i] = res.x

        return theta_vec

    def output_sigma_n_det(self, theta):
        '''
        Compute sigma_n_det at the measurement times for a given (theta_k, theta_D).

        Args:
            theta (array-like): 2-element array [theta_k, theta_D].

        Returns:
            numpy.ndarray: Deterministic sigma_n for each measurement time.
        '''
        if self.meas_time is None:
            raise ValueError('meas_time is not set.')

        return self.sigma_n_det_theta(theta, self.meas_time)

    def compute_std_meas(self):
        '''
        Estimate the standard deviation of the measurement noise by comparing
        the measured data to the model's deterministic prediction at (theta_k=1, theta_D=1).

        Returns:
            float: Computed standard deviation.
        '''
        if self.meas_time is None or self.meas_sigma_n is None:
            raise ValueError('meas_time and meas_sigma_n must be set.')

        # Evaluate sigma_n_det at (1,1)
        theta_ref = np.array([1.0, 1.0])
        sigma_n_det = self.output_sigma_n_det(theta_ref)

        loc_meas_sigma_n = self.meas_sigma_n
        residual = sigma_n_det - loc_meas_sigma_n
        computed_variance = (1.0 / residual.size) * np.sum(residual**2)
        computed_std = np.sqrt(computed_variance)

        self.standard_dev = computed_std
        return computed_std

    # -------------------------------------------------------------------------
    # Protected Methods (Internal usage)
    # -------------------------------------------------------------------------
    def _dist_sigma_n_sigma_n_meas_log(self, log10_theta):
        '''
        Compute the squared norm of residuals between measured sigma_n and
        sigma_n_det, where (theta_k, theta_D) = 10^(log10_theta).

        Args:
            log10_theta (array-like): log10 of the parameter vector [theta_k, theta_D].

        Returns:
            float: The squared error.
        '''
        if self.meas_time is None or self.meas_sigma_n is None:
            raise ValueError('meas_time and meas_sigma_n must be set.')

        theta = 10.0**log10_theta
        sigma_n_est = self.sigma_n_det_theta(theta, self.meas_time)
        residual = sigma_n_est - self.meas_sigma_n
        return np.sum(residual**2)

    def _dist_sigma_n_sigma_n_meas(self, theta):
        '''
        Compute the squared norm of residuals between measured sigma_n and
        sigma_n_det for a given (theta_k, theta_D).

        Args:
            theta (array-like): [theta_k, theta_D] in linear space.

        Returns:
            float: The squared error.
        '''
        if self.meas_time is None or self.meas_sigma_n is None:
            raise ValueError('meas_time and meas_sigma_n must be set.')

        sigma_n_est = self.sigma_n_det_theta(theta, self.meas_time)
        residual = sigma_n_est - self.meas_sigma_n
        return np.sum(residual**2)