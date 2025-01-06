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
        horizon: int,
        identified_shocks: np.ndarray = None,
        var_order: int = 0,
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
        self.horizon = horizon
        self.var_order = var_order

        # Shape
        self.num_obs, self.num_var = self.data.shape
        self.T = self.num_obs - 1 - self.horizon - self.var_order

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
            self.X = self.data.iloc[var_order : self.T + var_order].to_numpy().T
        else:
            self.X = self.data[self.controls].iloc[: self.T].to_numpy().T
            self.Z = self.data[self.instruments].iloc[1 : self.T + 1].to_numpy().T
        self.y_H = np.column_stack(
            [
                self.data[self.target].iloc[i : i + self.T].to_numpy()
                for i in range(1, self.horizon + 1)
            ]
        ).T
        self.Y_H = np.hstack(
            [
                np.column_stack(
                    [
                        self.data[col].iloc[i : i + self.T].to_numpy()
                        for i in range(1, self.horizon + 1)
                    ]
                )
                for col in self.regressors
            ]
        ).T

    def proj_controls(self):
        """Calculates the projection matrix for controls."""
        inv = np.linalg.inv(self.X @ self.X.T)
        self.M_X = np.eye(self.T) - self.X.T @ inv @ self.X

    def perp_matrix_lp(self):
        """Calculates the perpendicular matrices for LP-IV."""
        return self.y_H @ self.M_X, self.Y_H @ self.M_X, self.Z @ self.M_X

    def init_var(self, var_list: list = None):
        """Initializes the VAR model.

        Args:
            var_list (list, optional): List of variables to include in the VAR. Defaults to None.
        """
        data = self.data if var_list is None else self.data[var_list]
        self.var = VAR(
            data.iloc[: self.T + self.var_order],
        )

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
        irfs = self.fitted_var.irf(
            periods=self.horizon, var_decomp=var_decomp, var_order=var_order
        )
        self.irfs = (
            irfs.orth_irfs @ self.identified_shocks
            if self.identified_shocks is not None
            else irfs
        )

    def irfs_lp(self):
        """Calculates the impulse response functions for LP-IV."""
        y_perp, Y_perp, Z_perp = self.perp_matrix()
        norm = sqrtm((Z_perp @ Z_perp.T) / self.T)
        return ((y_perp @ Z_perp.T) / self.T) @ norm, (
            (Y_perp @ Z_perp.T) / self.T
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
        return (
            struct_shocks @ self.identified_shocks
            if self.identified_shocks is not None
            else Z
        )

    def perp_matrix_var(self):
        """Calculates the perpendicular matrices for VAR."""
        self.fitted_var.forecats
        pass

    def transform_irf(self, function):
        """Transforms the impulse response functions.

        Args:
            function: Transformation function.
        """
        pass

    def regression_irf(self, impulse: str, response: list, keep_moments: list):
        """Performs regression on the impulse response functions.

        Args:
            impulse (str): Name of the impulse variable.
            response (str): Name of the response variable.
        """
        pass

    def get_klm(self):
        """Calculates the KLM statistic."""
        pass
