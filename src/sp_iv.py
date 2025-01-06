import pandas as pd
import numpy as np
import statsmodels.api as sm

from statsmodels.tsa.api import VAR
from scipy.linalg import sqrtm


class SP_IV:

    def __init__(
        self,
        data: pd.DataFrame,
        order_dict: dict,
        spec: str,
        horizon: int,
        identified_shocks: np.ndarray = None,
        var_order: int = 0,
    ):
        """initialize class

        Args:
            data (pd.DataFrame): Data stored as pd.Dataframe with their id.
            dict (dict): Key are series name ordered, Items are list to encode if instruments, controls, regressors or target.
            spec (str): Either "LP" or "VAR" for the forecasting step.
        """

        # Data
        self.data = data[list(order_dict.keys())]
        self.order_dict = order_dict
        self.instruments, self.controls, self.regressors = [], [], []

        for key, value in order_dict.items():
            if "instruments" in value:
                self.instruments.append(
                    key
                )  # for VAR if an identified shocks is obtained from a linear combination of all variables then instruments should include all those
            elif "controls" in value:
                self.controls.append(key)
            elif "regressors" in value:
                self.regressors.append(key)
            elif "target" in value:
                self.target = key

        # Spec
        self.spec = spec
        self.horizon = horizon

        # Shape
        self.num_obs, self.num_var = self.data.iloc[1:].shape
        self.T = self.num_obs - 1 - self.horizon - self.order
        self.var_order = var_order

        self.N_z = len(self.instruments)
        self.N_Y = len(self.regressors)
        self.N_x = len(self.controls) if spec == "LP" else self.N

        # Matrix
        self.init_matrix(var_order=var_order)

        # Identified Shock
        self.identified_shocks = identified_shocks

    def init_matrix(self, var_order: int):
        if self.spec == "VAR":
            self.X = self.data.iloc[var_order : self.T + var_order].to_numpy().T
        else:
            self.X = self.data[self.controls].iloc[: self.T].to_numpy().T
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
        self.Z = self.data[self.instruments].iloc[1 : self.T + 1].to_numpy().T

    def proj_controls(self):
        inv = np.linalg.inv(self.X @ self.X.T)
        self.M_X = np.eye(self.T) - self.X.T @ inv @ self.X

    def perp_matrix_lp(self):
        return self.y_H @ self.M_X, self.Y_H @ self.M_X, self.Z @ self.M_X

    def init_var(self, var_list: list = None):
        data = self.data if var_list is None else self.data[var_list]
        self.model = VAR(
            data.iloc[: self.T + self.var_order],
        )

    def fit_var(self, order: int = None, trend: str = "n"):
        self.fitted_model = self.model.fit(maxlags=order, trend=trend)

    def irfs_var(
        self,
        var_decomp: np.ndarray = None,
        var_order=None,
    ):
        irfs = self.fitted_model.irf(
            periods=self.horizon, var_decomp=var_decomp, var_order=var_order
        )  # Shape: (forecast_steps, num_variables, num_variables)
        self.irfs = (
            irfs @ self.identified_shocks
            if self.identified_shocks is not None
            else irfs.orth_irfs
        )

    def plot_var_irf(
        self, impulse: str, figsize: tuple = (5, 20), signif: float = 0.68, orth=True
    ):
        if self.identified_shocks is None:
            self.irfs.plot(orth=orth, impulse=impulse, figsize=figsize, signif=signif)

    def irfs_lp(self):
        y_perp, Y_perp, Z_perp = self.perp_matrix()
        norm = sqrtm((Z_perp @ Z_perp.T) / self.T)
        return ((y_perp @ Z_perp.T) / self.T) @ norm, (
            (Y_perp @ Z_perp.T) / self.T
        ) @ norm

    def perp_instruments_var(self):

        residuals = self.fitted_model.resid
        Z_perp = (
            residuals[self.instruments]
            .iloc[self.var_order + 1 : self.var_order + self.T + 1]
            .T
        )  # instruments are the series you add as source of shock for example in Ramey (2011), instruments are war dates and z_t^perp = shock in war_dates, is identified because no other variables impact it and we assume cholesky decomposition.
        return (
            self.identified_shocks @ Z_perp
            if self.identified_shocks is not None
            else Z_perp
        )  # TODO what is Z in the MBC shocks, since a shock already orthonal so Z = Z_perp and Z = MBC_shock or use the serie used to identify in the MBC, unemployment no The MBC shock itself, as identified in the VAR, is the proxy (or instrument) used in the estimation of the Phillips curve. ?

    def perp_matrix_var(self):
        pass

    def transform_irf(self, function):
        pass

    def regression_irf(self, impulse: str, response: str):
        pass

    def get_klm(self):
        pass
