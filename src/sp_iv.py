import pandas as pd
import numpy as np
import statsmodels.api as sm

from statsmodels.tsa.api import VAR
from scipy.linalg import sqrtm


class SP_IV:

    def __init__(self, data: pd.DataFrame, order_dict: dict, spec: str, horizon: int):
        """initialize class

        Args:
            data (pd.DataFrame): Data stored as pd.Dataframe with their id.
            dict (dict): Key are series name ordered, Items encode if instruments, controls, regressors or target.
            spec (str): Either "LP" or "VAR" for the forecasting step.
        """

        # Data
        self.data = data[list(order_dict.keys())]
        self.order_dict = order_dict
        self.instruments, self.controls, self.regressors = [], [], []

        for key, value in order_dict.items():
            if value == "instruments":
                self.instruments.append(key)
            elif value == "controls":
                self.controls.append(key)
            elif value == "regressors":
                self.regressors.append(key)
            elif value == "target":
                self.target = key

        # Spec
        self.spec = spec
        self.horizon = horizon

        # Shape
        self.num_obs, self.N = self.data.iloc[1:].shape
        self.T = self.num_obs - 1 - self.horizon

        self.N_z = len(self.instruments)
        self.N_Y = len(self.regressors)
        self.N_x = len(self.controls) if spec == "LP" else self.N

    def get_matrix(self):
        if self.spec == "VAR":
            self.X = self.data.iloc[: self.T].to_numpy().T
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

    def perp_matrix(self):
        return self.y_H @ self.M_X, self.Y_H @ self.M_X, self.Z @ self.M_X

    def init_model(self):
        if self.spec == "VAR":
            self.model = VAR(self.data.iloc[1:])

    def fit_model(self, order: int = None):
        if self.spec == "VAR":
            self.fitted_model = self.model.fit(order)

    def compute_irfs(self):
        y_perp, Y_perp, Z_perp = self.perp_matrix()
        if self.spec == "LP":
            norm = sqrtm((Z_perp @ Z_perp.T) / self.T)
            return ((y_perp @ Z_perp.T) / self.T) @ norm, (
                (Y_perp @ Z_perp.T) / self.T
            ) @ norm
        else:
            return

    def transform_irf(self, function):
        pass

    def regression_irf(self):
        pass

    def get_klm(self):
        pass
