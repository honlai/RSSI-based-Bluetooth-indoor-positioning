import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter
from filterpy.kalman import ExtendedKalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints

class KalmanFilterSet:
    """A set of independent 1D Kalman filters, one for each RSSI field.

    Each field is filtered individually without modeling interaction between them.
    This is suitable when RSSI observations are assumed to be uncorrelated.
    """

    def __init__(self, initial_values: pd.Series, r_val=4, q_val=0.01, p_val=10):
        """Initializes the KalmanFilterSet.

        Args:
            initial_values (pd.Series): Initial RSSI values for each field.
            r_val (float): Measurement noise covariance R. Defaults to 4.
            q_val (float): Process noise covariance Q. Defaults to 0.01.
            p_val (float): Initial estimate covariance P. Defaults to 10.

        Example:
            >>> initial = pd.Series({'A': -70.0, 'B': -75.0})
            >>> kf_set = KalmanFilterSet(initial)
        """
        self.name = 'kalman_filter_set'
        self.columns = initial_values.index.tolist()
        self.n = len(self.columns)
        self.filters = {}

        for col, init_val in initial_values.items():
            kf = KalmanFilter(dim_x=1, dim_z=1)
            kf.x = np.array([[init_val]])
            kf.F = np.array([[1]])
            kf.H = np.array([[1]])
            kf.P *= p_val
            kf.R = r_val
            kf.Q = q_val
            self.filters[col] = kf

    def update(self, new_row: pd.Series) -> pd.Series:
        """Updates all filters with new RSSI measurements.

        Args:
            new_row (pd.Series): New RSSI measurements for each field.

        Returns:
            pd.Series: Filtered RSSI values.
        
        Example:
            >>> rssi = pd.Series({'A': -69.5, 'B': -74.2})
            >>> filtered = kf_set.update(rssi)
        """
        filtered_result = {}
        for col, z in new_row.items():
            kf = self.filters[col]
            kf.predict()
            kf.update(np.array([[z]]))
            filtered_result[col] = kf.x[0, 0]
        return pd.Series(filtered_result)

class MultiKalmanFilter:
    """A multi-dimensional Kalman filter for correlated RSSI fields.

    This filter models the full RSSI vector as a joint state, capturing any potential
    correlation between sources or measurement noise.
    """

    def __init__(self, initial_values: pd.Series, r_val=4, q_val=0.01, p_val=10):
        """Initializes the multi-dimensional Kalman filter.

        Args:
            initial_values (pd.Series): Initial RSSI values for each field.
            r_val (float): Measurement noise covariance. Defaults to 4.
            q_val (float): Process noise covariance. Defaults to 0.01.
            p_val (float): Initial estimate covariance. Defaults to 10.

        Example:
            >>> initial = pd.Series({'A': -65.0, 'B': -70.0})
            >>> mkf = MultiKalmanFilter(initial)
        """
        self.name = 'multi_kalman_filter'
        self.columns = initial_values.index.tolist()
        self.n = len(self.columns)

        self.kf = KalmanFilter(dim_x=self.n, dim_z=self.n)
        self.kf.x = initial_values.values.reshape(self.n, 1)
        self.kf.F = np.eye(self.n)
        self.kf.H = np.eye(self.n)
        self.kf.P *= p_val
        self.kf.R = np.eye(self.n) * r_val
        self.kf.Q = np.eye(self.n) * q_val

    def update(self, new_row: pd.Series) -> pd.Series:
        """Updates the Kalman filter with a new RSSI observation.

        Args:
            new_row (pd.Series): New RSSI values for each field.

        Returns:
            pd.Series: Filtered RSSI values.

        Example:
            >>> rssi = pd.Series({'A': -64.8, 'B': -69.9})
            >>> filtered = mkf.update(rssi)
        """
        z = new_row.values.reshape(self.n, 1)
        self.kf.predict()
        self.kf.update(z)
        return pd.Series(self.kf.x.flatten(), index=self.columns)

class KalmanWithVelocitySet:
    """A set of 2D Kalman filters for tracking RSSI and its velocity per field.

    Each filter models the system state as [RSSI, dRSSI], using a constant velocity model.
    Useful for smoothing RSSI signals and estimating their trends over time.
    """

    def __init__(self, initial_values: pd.Series, r_val=4, q_val=0.01, p_val=10):
        """Initializes a Kalman filter for each field with [RSSI, dRSSI] state.

        Args:
            initial_values (pd.Series): Initial RSSI values for each field.
            r_val (float): Measurement noise covariance R. Defaults to 4.
            q_val (float): Process noise covariance Q. Defaults to 0.01.
            p_val (float): Initial estimate covariance P. Defaults to 10.

        Example:
            >>> initial = pd.Series({'A': -65.0, 'B': -70.0})
            >>> kf_vel = KalmanWithVelocitySet(initial)
        """
        self.name = 'kalman_with_velocity_set'
        self.columns = initial_values.index.tolist()
        self.n = len(self.columns)

        self.filters = {}
        for col, init_val in initial_values.items():
            kf = KalmanFilter(dim_x=2, dim_z=1)
            kf.x = np.array([[init_val], [0]])        # Initial RSSI and velocity
            kf.F = np.array([[1, 1], [0, 1]])          # State transition model
            kf.H = np.array([[1, 0]])                  # Measurement model
            kf.P *= p_val
            kf.R = r_val
            kf.Q = np.array([[q_val, 0], [0, q_val]])  # Process noise covariance
            self.filters[col] = kf

    def update(self, new_row: pd.Series) -> pd.Series:
        """Updates each filter with new RSSI measurements and returns filtered RSSI.

        The velocity component is not returned.

        Args:
            new_row (pd.Series): Observed RSSI values per field.

        Returns:
            pd.Series: Filtered RSSI values.

        Example:
            >>> rssi = pd.Series({'A': -64.5, 'B': -71.3})
            >>> filtered_rssi = kf_vel.update(rssi)
        """
        result = {}
        for col, z in new_row.items():
            kf = self.filters[col]
            kf.predict()
            kf.update(np.array([[z]]))
            result[col] = kf.x[0, 0]  # Return RSSI only (not velocity)
        return pd.Series(result)

class ExtendedKalmanFilterSet:
    """A set of Extended Kalman Filters (EKF) for tracking RSSI per field.

    Each filter assumes constant RSSI dynamics (i.e., no change in prediction).
    Suitable for systems with weak non-linearity or stable signal conditions.
    """

    def __init__(self, initial_values: pd.Series, r_val=4, q_val=0.01, p_val=10):
        """Initializes an EKF for each field assuming constant RSSI dynamics.

        Args:
            initial_values (pd.Series): Initial RSSI values for each field.
            r_val (float): Measurement noise covariance R. Defaults to 4.
            q_val (float): Process noise covariance Q. Defaults to 0.01.
            p_val (float): Initial estimate covariance P. Defaults to 10.

        Example:
            >>> init_rssi = pd.Series({'A': -60.0, 'B': -70.0})
            >>> ekf = ExtendedKalmanFilterSet(init_rssi)
        """
        self.name = 'extended_kalman_filter_set'
        self.columns = initial_values.index.tolist()
        self.n = len(self.columns)

        self.filters = {}
        for col, init_val in initial_values.items():
            ekf = ExtendedKalmanFilter(dim_x=1, dim_z=1)
            ekf.x = np.array([[init_val]])
            ekf.P *= p_val
            ekf.R = np.array([[r_val]])
            ekf.Q = np.array([[q_val]])
            self.filters[col] = ekf

    @staticmethod
    def fx(x, dt=1.0):
        """State transition function assuming constant RSSI.

        Args:
            x (np.ndarray): Current state.
            dt (float): Time step. Defaults to 1.0.

        Returns:
            np.ndarray: Predicted next state.
        """
        return np.array([[x[0]]])

    @staticmethod
    def hx(x):
        """Measurement function that returns RSSI.

        Args:
            x (np.ndarray): Current state.

        Returns:
            np.ndarray: Expected measurement.
        """
        return np.array([[x[0, 0]]])

    @staticmethod
    def F_jacobian(x, dt=1.0):
        """Jacobian of the state transition function.

        Args:
            x (np.ndarray): Current state.
            dt (float): Time step. Defaults to 1.0.

        Returns:
            np.ndarray: State transition Jacobian matrix.
        """
        return np.array([[1]])

    @staticmethod
    def H_jacobian(x):
        """Jacobian of the measurement function.

        Args:
            x (np.ndarray): Current state.

        Returns:
            np.ndarray: Measurement function Jacobian matrix.
        """
        return np.array([[1]])

    def update(self, new_row: pd.Series) -> pd.Series:
        """Updates each EKF with a new RSSI measurement.

        Performs prediction and update using the assumption of constant RSSI.

        Args:
            new_row (pd.Series): Observed RSSI values per field.

        Returns:
            pd.Series: Filtered RSSI estimates.

        Example:
            >>> rssi = pd.Series({'A': -59.2, 'B': -71.5})
            >>> filtered = ekf.update(rssi)
        """
        result = {}
        for col, z in new_row.items():
            ekf = self.filters[col]
            ekf.F = self.F_jacobian(ekf.x)
            ekf.H = self.H_jacobian(ekf.x)
            ekf.predict()
            ekf.update(np.array([[z]]), HJacobian=self.H_jacobian, Hx=self.hx)
            result[col] = ekf.x[0, 0]
        return pd.Series(result)

class ExtendedKalmanFilterSetNonlinear:
    """A set of nonlinear Extended Kalman Filters (EKFs) using an exponential measurement model.

    This model assumes observed values (e.g., distance) are exponential functions of RSSI:
        z = exp(-RSSI)

    It is suitable when signal attenuation is modeled exponentially, such as RSSI-to-distance mappings.
    """

    def __init__(self, initial_values: pd.Series, r_val=1.0, q_val=0.01, p_val=10):
        """Initializes one nonlinear EKF per field with exponential observation model.

        Args:
            initial_values (pd.Series): Initial RSSI values for each field.
            r_val (float): Measurement noise covariance R. Default is 1.0.
            q_val (float): Process noise covariance Q. Default is 0.01.
            p_val (float): Initial estimate covariance P. Default is 10.

        Example:
            >>> init_rssi = pd.Series({'A': -55.0, 'B': -70.0})
            >>> ekf = ExtendedKalmanFilterSetNonlinear(init_rssi)
        """
        self.name = 'extended_kalman_filter_set_nonlinear'
        self.columns = initial_values.index.tolist()
        self.n = len(self.columns)

        self.filters = {}
        for col, init_val in initial_values.items():
            ekf = ExtendedKalmanFilter(dim_x=1, dim_z=1)
            ekf.x = np.array([[init_val]])
            ekf.P *= p_val
            ekf.R = np.array([[r_val]])
            ekf.Q = np.array([[q_val]])
            self.filters[col] = ekf

    @staticmethod
    def hx(x):
        """Nonlinear measurement function: z = exp(-RSSI).

        Args:
            x (np.ndarray): Current state vector [RSSI].

        Returns:
            np.ndarray: Expected measurement z.
        """
        return np.array([[np.exp(-x[0, 0])]])

    @staticmethod
    def H_jacobian(x):
        """Jacobian of the measurement function z = exp(-x).

        Args:
            x (np.ndarray): Current state vector [RSSI].

        Returns:
            np.ndarray: Measurement function Jacobian matrix.
        """
        return np.array([[-np.exp(-x[0, 0])]])

    @staticmethod
    def F_jacobian(x, dt=1.0):
        """Jacobian of the state transition function (identity).

        Args:
            x (np.ndarray): Current state vector [RSSI].
            dt (float): Time step. Defaults to 1.0.

        Returns:
            np.ndarray: State transition Jacobian matrix.
        """
        return np.array([[1]])

    def update(self, new_row: pd.Series) -> pd.Series:
        """Updates each EKF using nonlinear measurement z = exp(-RSSI).

        Args:
            new_row (pd.Series): Observed values (e.g., distances) for each field.

        Returns:
            pd.Series: Estimated RSSI values after filtering.

        Example:
            >>> distances = pd.Series({'A': 0.01, 'B': 0.0005})
            >>> rssi_estimates = ekf.update(distances)
        """
        result = {}
        for col, z in new_row.items():
            ekf = self.filters[col]
            ekf.F = self.F_jacobian(ekf.x)
            ekf.predict()
            ekf.update(np.array([[z]]), HJacobian=self.H_jacobian, Hx=self.hx)
            result[col] = ekf.x[0, 0]
        return pd.Series(result)

class ExtendedKalmanFilterWithVelocitySet:
    """Extended Kalman Filter (EKF) set with constant velocity model for RSSI.

    Each EKF tracks both RSSI and its rate of change (dRSSI) using a linear motion model.
    Only RSSI is observed; velocity is estimated implicitly.

    State vector for each filter:
        [RSSI, dRSSI]
    """

    def __init__(self, initial_values: pd.Series, dt=1.0, r_val=4,
                 q_rssi=0.01, q_velocity=0.01, p_val=10):
        """Initializes a set of EKFs with [RSSI, dRSSI] state per field.

        Args:
            initial_values (pd.Series): Initial RSSI values per field.
            dt (float): Time step between updates. Default is 1.0.
            r_val (float): Measurement noise covariance R. Default is 4.
            q_rssi (float): Process noise for RSSI state. Default is 0.01.
            q_velocity (float): Process noise for dRSSI state. Default is 0.01.
            p_val (float): Initial estimate covariance P. Default is 10.

        Example:
            >>> init_rssi = pd.Series({'A': -70.0, 'B': -65.0})
            >>> ekf_set = ExtendedKalmanFilterWithVelocitySet(init_rssi)
        """
        self.name = 'extended_kalman_filter_with_velocity_set'
        self.columns = initial_values.index.tolist()
        self.n = len(self.columns)
        self.dt = dt

        self.filters = {}
        for col, init_val in initial_values.items():
            ekf = ExtendedKalmanFilter(dim_x=2, dim_z=1)
            ekf.x = np.array([[init_val], [0.0]])
            ekf.P = np.eye(2) * p_val
            ekf.R = np.array([[r_val]])
            ekf.Q = np.array([
                [q_rssi, 0],
                [0, q_velocity]
            ])
            self.filters[col] = ekf

    def fx(self, x, dt):
        """State transition function assuming constant velocity.

        Args:
            x (np.ndarray): Current state vector [RSSI, dRSSI].
            dt (float): Time step.

        Returns:
            np.ndarray: Predicted next state [RSSI + dRSSI * dt, dRSSI].
        """
        rssi = x[0, 0] + x[1, 0] * dt
        drssi = x[1, 0]
        return np.array([[rssi], [drssi]])

    def hx(self, x):
        """Measurement function returning only RSSI component.

        Args:
            x (np.ndarray): Current state vector [RSSI, dRSSI].

        Returns:
            np.ndarray: Observed RSSI value.
        """
        return np.array([[x[0, 0]]])

    def F_jacobian(self, x, dt):
        """Jacobian of the state transition function fx.

        Args:
            x (np.ndarray): Current state vector.
            dt (float): Time step.

        Returns:
            np.ndarray: 2x2 Jacobian matrix.
        """
        return np.array([
            [1, dt],
            [0, 1]
        ])

    def H_jacobian(self, x):
        """Jacobian of the measurement function hx.

        Args:
            x (np.ndarray): Current state vector.

        Returns:
            np.ndarray: 1x2 Jacobian matrix.
        """
        return np.array([[1, 0]])

    def update(self, rssi_row: pd.Series, with_velocity: bool = False
               ) -> pd.Series | tuple[pd.Series, pd.Series]:
        """Updates each EKF with a new RSSI observation.

        Predicts and updates the EKF using observed RSSI values. Optionally returns velocity.

        Args:
            rssi_row (pd.Series): RSSI measurements for each field.
            with_velocity (bool): If True, return both RSSI and dRSSI estimates.

        Returns:
            pd.Series: Filtered RSSI values (if with_velocity is False).
            tuple[pd.Series, pd.Series]: (RSSI, dRSSI) estimates (if with_velocity is True).

        Example:
            >>> obs = pd.Series({'A': -69.2, 'B': -66.1})
            >>> filtered = ekf_set.update(obs)
            >>> rssi, drssi = ekf_set.update(obs, with_velocity=True)
        """
        result = {}
        velocity = {}

        for col, z in rssi_row.items():
            ekf = self.filters[col]
            ekf.F = self.F_jacobian(ekf.x, self.dt)
            ekf.predict(fx=self.fx, Fx=self.F_jacobian, args=(self.dt,))
            ekf.update(np.array([[z]]), HJacobian=self.H_jacobian, Hx=self.hx)
            result[col] = ekf.x[0, 0]
            velocity[col] = ekf.x[1, 0]

        if with_velocity:
            return pd.Series(result), pd.Series(velocity)
        else:
            return pd.Series(result)

class UKFWithVelocity:
    """Unscented Kalman Filter for RSSI and its velocity (dRSSI).

    The filter estimates a 2D state vector:
        [RSSI, dRSSI]

    It applies the Unscented Transform to handle nonlinear dynamics without requiring Jacobians.
    Suitable for time-varying RSSI with smooth changes in velocity.

    Example:
        >>> ukf = UKFWithVelocity(initial_rssi=-60.0)
        >>> rssi = ukf.update(observed_rssi=-62.3)
        >>> rssi, velocity = ukf.update(observed_rssi=-61.8, with_velocity=True)
    """

    def __init__(self, initial_rssi: float, initial_velocity: float = 0.0,
                 dt=1.0, r_val=4.0, q_rssi=0.01, q_velocity=0.01, p_val=10):
        """Initializes the UKF state and noise parameters.

        Args:
            initial_rssi (float): Initial RSSI value.
            initial_velocity (float, optional): Initial RSSI rate of change. Defaults to 0.0.
            dt (float, optional): Time step between updates. Defaults to 1.0.
            r_val (float, optional): Measurement noise covariance R. Defaults to 4.0.
            q_rssi (float, optional): Process noise for RSSI. Defaults to 0.01.
            q_velocity (float, optional): Process noise for velocity. Defaults to 0.01.
            p_val (float, optional): Initial state covariance. Defaults to 10.
        """
        self.name = 'unscented_kalman_filter_with_velocity'
        self.columns = None
        self.n = 1

        self.dt = dt
        self.points = MerweScaledSigmaPoints(n=2, alpha=0.1, beta=2.0, kappa=0)

        self.ukf = UnscentedKalmanFilter(
            dim_x=2,
            dim_z=1,
            dt=dt,
            hx=self.hx,
            fx=self.fx,
            points=self.points
        )

        self.ukf.x = np.array([initial_rssi, initial_velocity])
        self.ukf.P *= p_val
        self.ukf.Q = np.diag([q_rssi, q_velocity])
        self.ukf.R = np.array([[r_val]])

    def fx(self, x, dt):
        """State transition function with constant velocity.

        Args:
            x (np.ndarray): Current state vector [RSSI, dRSSI].
            dt (float): Time step.

        Returns:
            np.ndarray: Predicted next state vector.
        """
        return np.array([x[0] + x[1] * dt, x[1]])

    def hx(self, x):
        """Measurement function observing only RSSI.

        Args:
            x (np.ndarray): Current state vector [RSSI, dRSSI].

        Returns:
            np.ndarray: Measurement vector [RSSI].
        """
        return np.array([x[0]])

    def update(self, observed_rssi: float, with_velocity: bool = False) -> float | tuple[float, float]:
        """Updates the UKF with an observed RSSI value.

        Args:
            observed_rssi (float): Observed RSSI value.
            with_velocity (bool, optional): Whether to return dRSSI estimate. Defaults to False.

        Returns:
            float: Estimated RSSI if `with_velocity` is False.
            tuple[float, float]: (RSSI, dRSSI) if `with_velocity` is True.
        """
        self.ukf.predict()
        self.ukf.update(np.array([observed_rssi]))
        if with_velocity:
            return self.ukf.x[0], self.ukf.x[1]
        else:
            return self.ukf.x[0]

class UKFWithVelocitySet:
    """A collection of UKFWithVelocity filters for multiple RSSI fields.

    Each RSSI field is tracked independently using its own Unscented Kalman Filter (UKF).
    The state for each filter is [RSSI, dRSSI], modeling signal dynamics with velocity.

    Example:
        >>> init_rssi = pd.Series({'A': -65.0, 'B': -70.0})
        >>> ukf_set = UKFWithVelocitySet(init_rssi)
        >>> obs = pd.Series({'A': -66.2, 'B': -69.8})
        >>> filtered = ukf_set.update(obs)
        >>> rssi, velocity = ukf_set.update(obs, with_velocity=True)
    """

    def __init__(self, initial_values: pd.Series,
                 dt=1.0, r_val=4.0, q_rssi=0.01, q_velocity=0.01, p_val=10):
        """Initializes a UKF for each RSSI field.

        Args:
            initial_values (pd.Series): Initial RSSI values for each field.
            dt (float, optional): Time step between updates. Defaults to 1.0.
            r_val (float, optional): Measurement noise covariance R. Defaults to 4.0.
            q_rssi (float, optional): Process noise for RSSI. Defaults to 0.01.
            q_velocity (float, optional): Process noise for velocity. Defaults to 0.01.
            p_val (float, optional): Initial state covariance. Defaults to 10.
        """
        self.name = 'unscented_kalman_filter_with_velocity_set'
        self.columns = initial_values.index.tolist()
        self.n = len(self.columns)

        self.filters = {}
        for col, init_val in initial_values.items():
            self.filters[col] = UKFWithVelocity(
                initial_rssi=init_val,
                dt=dt,
                r_val=r_val,
                q_rssi=q_rssi,
                q_velocity=q_velocity,
                p_val=p_val
            )

    def update(self, rssi_row: pd.Series, with_velocity: bool = False) -> pd.Series | tuple[pd.Series, pd.Series]:
        """Updates all UKFs with new RSSI measurements.

        Args:
            rssi_row (pd.Series): Observed RSSI values for each field.
            with_velocity (bool, optional): If True, returns velocity estimates.

        Returns:
            pd.Series: Filtered RSSI values if `with_velocity` is False.
            tuple[pd.Series, pd.Series]: (RSSI, dRSSI) values if `with_velocity` is True.
        """
        result = {}
        velocity = {}

        for col, value in rssi_row.items():
            r, v = self.filters[col].update(value, with_velocity=True)
            result[col] = r
            velocity[col] = v

        if with_velocity:
            return pd.Series(result), pd.Series(velocity)
        else:
            return pd.Series(result)
