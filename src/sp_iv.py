import pandas as pd
import numpy as np
import statsmodels.api as sm

from statsmodels.tsa.api import VAR
from scipy.linalg import sqrtm, cholesky

from utils.plot import plot_irfs


class SP_IV:
    """Implements SVAR-IV and LP-IV estimation.

    This class provides methods for estimating impulse response functions using
    SVAR-IV and LP-IV approaches. It handles data preparation,
    VAR fitting, IRF calculation, and plotting.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        order_dict: dict,
        spec: str,
        horizons: range,
        identified_shocks: np.ndarray = None,
        var_order: int = 0,
        burn_in: int = 0,
    ):
        """Initializes the SP_IV class.

        Args:
            data (pd.DataFrame): Dataframe containing the time series data.
            order_dict (dict): Dictionary specifying the role of each variable
                (instruments, controls, regressors, target).
            spec (str): Specification of the forecasting step ("LP" or "VAR").
            horizon (int): Forecast horizon.
            identified_shocks (np.ndarray, optional): Identified shocks. Defaults to None.
            var_order (int, optional): VAR order. Defaults to 0.
        """

        # Data
        self.data = data[list(order_dict.keys())]
        self.order_dict = order_dict
        self.instruments, self.controls, self.regressors = [], [], []

        for key, value in order_dict.items():
            if "instruments" in value:
                self.instruments.append(key)
            if "controls" in value:
                self.controls.append(key)
            elif "regressors" in value:
                self.regressors.append(key)
            elif "target" in value:
                self.target = key

        # Spec
        self.spec = spec
        self.horizons = horizons
        self.horizon = max(horizons)
        self.var_order = var_order
        self.burn_in = burn_in

        # Shape
        self.num_obs, self.num_var = self.data.shape
        self.start_time = self.var_order + self.burn_in
        self.T = self.num_obs - 1 - self.horizon - self.var_order - self.burn_in

        self.N_z = len(self.instruments)
        self.N_Y = len(self.regressors)
        self.N_x = len(self.controls) if spec == "LP" else self.num_var

        # Matrix
        self.init_matrix(var_order=var_order)

        # Identified Shock
        self.identified_shocks = identified_shocks

    def init_matrix(self, var_order: int):
        """Initializes the data matrices.

        Args:
            var_order (int): VAR order.
        """
        if self.spec == "VAR":
            self.X = np.vstack(
                [
                    self.data.iloc[
                        self.start_time - lag : self.T + self.start_time - lag
                    ]
                    .to_numpy()
                    .T
                    for lag in range(var_order)
                ]
            )
        else:
            self.X = self.data[self.controls].iloc[: self.T].to_numpy().T
            self.Z = self.data[self.instruments].iloc[1 : self.T + 1].to_numpy().T
        self.y_H = np.column_stack(
            [
                self.data[self.target]
                .iloc[h + self.start_time + 1 : h + self.T + self.start_time + 1]
                .to_numpy()
                for h in self.horizons
            ]
        ).T
        self.Y_H = np.hstack(
            [
                np.column_stack(
                    [
                        self.data[col]
                        .iloc[
                            h + self.start_time + 1 : h + self.T + self.start_time + 1
                        ]
                        .to_numpy()
                        for h in self.horizons
                    ]
                )
                for col in self.regressors
            ]
        ).T
        inv_X = np.linalg.inv(self.X @ self.X.T)
        self.M_X = np.eye(self.T) - self.X.T @ inv_X @ self.X

    def perp_matrix_lp(self):
        """Calculates the perpendicular matrices for LP-IV."""
        self.y_perp = self.y_H @ self.M_X
        self.Y_perp = self.Y_H @ self.M_X
        self.Z_perp = self.Z @ self.M_X

    def init_var(self, var_list: list = None):
        """Initializes the VAR model.

        Args:
            var_list (list, optional): List of variables to include in the VAR. Defaults to None.
        """
        data = self.data if var_list is None else self.data[var_list]
        self.var = VAR(data)

    def fit_var(self, order: int = None, trend: str = "n"):
        """Fits the VAR model.

        Args:
            order (int, optional): VAR order. Defaults to None.
            trend (str, optional): Trend type. Defaults to "n".
        """
        self.fitted_var = self.var.fit(maxlags=order, trend=trend)

    def irfs_var(
        self,
        var_decomp: np.ndarray = None,
        var_order=None,
    ):
        """Calculates the impulse response functions for VAR.

        Args:
            var_decomp (np.ndarray, optional): Variance decomposition matrix. Defaults to None.
            var_order (optional): order of the decomposition matrix. Defaults to None.
        """
        irfs = self.fitted_var.irf(periods=self.horizon)
        self.irfs = (
            irfs.orth_irfs @ self.identified_shocks
            if self.identified_shocks is not None
            else irfs
        )

    def irfs_lp(self):
        """Calculates the impulse response functions for LP-IV."""
        norm = sqrtm((self.Z_perp @ self.Z_perp.T) / self.T)
        return ((self.y_perp @ self.Z_perp.T) / self.T) @ norm, (
            (self.Y_perp @ self.Z_perp.T) / self.T
        ) @ norm

    def instruments_var(self):
        """Calculates the instruments for VAR."""
        residuals = self.fitted_var.resid
        Z = (
            residuals[self.instruments]
            .iloc[self.var_order + 1 : self.var_order + self.T + 1]
            .T
        ).to_numpy()
        cov = self.fitted_var.resid_acov()[0]
        cholesky_cov = cholesky(cov, lower=True)
        struct_shocks = (np.linalg.inv(cholesky_cov) @ Z).T

        self.Z = (
            (struct_shocks @ self.identified_shocks).T
            if self.identified_shocks is not None
            else Z.T
        )

    def first_stage_error(self):
        XZ = np.vstack([self.X, self.Z])
        PXZ = XZ.T @ np.linalg.inv(XZ @ XZ.T) @ XZ
        MXZ = np.eye(self.T) - PXZ
        return MXZ @ self.Y_H.T  # v_T^perp

    def h_ahead_forecast_var(self, y_index: int, Y_indices: range, history: int = None):
        """Calculates the perpendicular matrices for VAR."""
        y_H_fcst = np.zeros((self.horizon, self.T + history))
        Y_H_fcst = np.zeros((self.horizon * self.N_Y, self.T + history))

        for h in range(self.horizon):
            for t in range(self.T + history):
                X_t = self.data.iloc[: t + self.start_time].values
                if h > history:
                    forecast = self.fitted_var.forecast(y=X_t, steps=h - history)[-1]
                    y_H_fcst[h, t] = forecast[y_index]  # y_t h-ahead forecast
                    Y_H_fcst[h * self.N_Y : (h + 1) * self.N_Y, t] = forecast[
                        Y_indices
                    ]  # Y_t h-ahead forecast
                else:
                    y_H_fcst[h, t] = X_t[-(history - h + 1), y_index]
                    Y_H_fcst[h * self.N_Y : (h + 1) * self.N_Y, t] = X_t[
                        -(history - h + 1), Y_indices
                    ]
        return y_H_fcst, Y_H_fcst

    def get_irfs(self):
        """Transforms the impulse response functions.

        Args:
            function: Transformation function.
        """
        return self.irfs

    def set_irfs(self, irfs):
        self.irfs = irfs

    def proj_instruments_lp(self):
        inv = np.linalg.inv(self.Z_perp @ self.Z_perp.T)
        self.M_Z_perp = self.Z_perp.T @ inv @ self.Z_perp

    def u_perp(self, b):
        return self.y_perp - (np.kron(b, np.eye(self.horizon)))

    def regression_irf(self, impulse_index: int, keep_moments: list):
        """Performs regression on the impulse response functions.

        Args:
            impulse (str): Name of the impulse variable.
            response (str): Name of the response variable.
        """
        y = self.irfs[:, list(self.order_dict.keys()).index(self.target), impulse_index]
        Y = np.hstack(
            [
                self.irfs[
                    :, list(self.order_dict.keys()).index(regressor), impulse_index
                ]
                for regressor in self.regressors
            ]
        )
        self.reg = sm.OLS(y, Y).fit()

    def get_klm(self, b: np.ndarray):
        """Calculates the KLM statistic."""
        pass
