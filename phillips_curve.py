# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 16:32:41 2024

@author: Maxime Coulet, Nina Stizi, Eliott Von-Pine
"""


# ==============================================================================
# Advanced Macroeconometrics - Final assignement
#
# Dynamic Identification Using System Projections on Instrumental Variables
# Working Paper No. 2204 – Dallas Fed
# ==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.api import VAR
import utils.preprocessing as prep
from scipy.io import savemat
from scipy.linalg import sqrtm, cholesky
from scipy.stats import chi2

from src.sp_iv import SP_IV

### Constructing MBC shock

# Load the data
file_path = "data/phillips_curve/monthly.csv"
data = pd.read_csv(file_path)

# Prepare data for VAR estimation
var_data, MBCshock_data = prep.clean_data(data)
# run these 2 below to save
# var_data.to_csv(r'data/var_data.csv',index=False)

# Convert DataFrame to a dictionary compatible with MATLAB

# Convert DataFrame to a NumPy array
data_matrix = MBCshock_data.to_numpy()

# Create a dictionary with the matrix and column names
matlab_dict = {"data": data_matrix, "columns": MBCshock_data.columns.to_list()}

# Save to .mat file
# savemat(r'data/MBC.mat', matlab_dict)
# MBCshock_data.to_csv(r'data/mbc_data.csv',index=False,header=False)

### REPLIC

# load mbc shock as linear combination of orthogonal shocks from Business Cycle Anatomy
file_path = "data/phillips_curve/MBC_shock2.csv"
MBCshock = pd.read_csv(file_path, header=None)

MBCshock = MBCshock.T

MBCshock_data["core_CPI_yoy"] = (
    MBCshock_data["core_CPI_annualised_percent_change"]
    .rolling(window=12)
    .apply(prep.compute_year_on_year, args=(100,))
)
MBCshock_data["core_CPI_yoy_lagged"] = MBCshock_data["core_CPI_yoy"].shift(1).fillna(0)
MBCshock_data["core_CPI_yoy_forward"] = MBCshock_data["core_CPI_yoy"].shift(-12)
MBCshock_data["y"] = (
    MBCshock_data["core_CPI_annualised_percent_change"]
    - MBCshock_data["core_CPI_yoy_lagged"]
)
MBCshock_data["gamma_f"] = (
    MBCshock_data["core_CPI_yoy_forward"] - MBCshock_data["core_CPI_yoy_lagged"]
)

MBCshock_data.dropna(axis=0, inplace=True)

order_dict = {
    "core_CPI_annualised_percent_change": ["target", "instruments"],
    "unemployment_rate": ["regressors", "instruments"],
    "log_industrial_production": ["controls", "instruments"],
    "PPI_commodity": ["controls", "instruments"],
    "10Y_Treasury_rate": ["controls", "instruments"],
    "3M_Treasury_rate": ["controls", "instruments"],
}

# Init model and irfs
horizons = range(0, 48, 3)
horizon = max(horizons) + 1
sp_iv = SP_IV(
    MBCshock_data[list(order_dict.keys())],
    order_dict=order_dict,
    spec="VAR",
    horizons=horizons,
    identified_shocks=MBCshock.to_numpy(),
    var_order=6,
    burn_in=12,
)
sp_iv.init_var()
sp_iv.fit_var(6, trend="n")
sp_iv.irfs_var()
sp_iv.instruments_var()

# Setup
H = 12
N_Y = 2
N_z = 1
N_X = 36
X, Z = sp_iv.X, sp_iv.Z
M_X = sp_iv.M_X
T = sp_iv.T
R = np.kron(np.eye(2), np.eye(H).flatten())

# Forecast
cpi_fcst, u_fcst = sp_iv.h_ahead_forecast_var(0, 1, 11)
u_fcst = u_fcst[:, 11:]
df_cpi_fcst = pd.DataFrame(data=cpi_fcst.T).iloc[11:]
df_cpi_yoy_fcst = (
    pd.DataFrame(data=cpi_fcst.T)
    .rolling(window=12)
    .apply(prep.compute_year_on_year, args=(100,))
    .dropna(axis=0)
)
y_H_perp = (
    (df_cpi_fcst - df_cpi_yoy_fcst.shift(1).fillna(0)).to_numpy().T[horizons[:-1]][:H]
)
cpi_H_perp = (
    (df_cpi_yoy_fcst.shift(-12).ffill() - df_cpi_yoy_fcst.shift(1).fillna(0))
    .to_numpy()
    .T[horizons[:-1]][:H]
)
u_H_perp = u_fcst[horizons[:-1]][:H]
Y_H_perp = np.vstack([cpi_H_perp, u_H_perp])

# IRFs
irfs = pd.DataFrame(
    data=sp_iv.irfs.reshape(horizon, 6),
    columns=[
        "core_CPI_annualised_percent_change",
        "unemployment_rate",
        "log_industrial_production",
        "PPI_commodity",
        "10Y_Treasury_rate",
        "3M_Treasury_rate",
    ],
)
irfs["core_CPI_yoy"] = (
    irfs["core_CPI_annualised_percent_change"]
    .rolling(window=12, min_periods=1)
    .apply(prep.compute_year_on_year)
)
irfs["core_CPI_yoy_lagged"] = irfs["core_CPI_yoy"].shift(1).fillna(0)
irfs["core_CPI_yoy_forward"] = irfs["core_CPI_yoy"].shift(-12)
irfs["y"] = irfs["core_CPI_annualised_percent_change"] - irfs["core_CPI_yoy_lagged"]
irfs["gamma_f"] = irfs["core_CPI_yoy_forward"] - irfs["core_CPI_yoy_lagged"]
irfs_q = irfs[irfs.index % 3 == 0]

# Plot the IRF
plt.figure(figsize=(12, 24))

plt.subplot(4, 1, 1)
plt.plot(
    irfs_q["core_CPI_annualised_percent_change"],
    label=r"IRF: $\pi^{1m}$ to 1-Std MBC Shock",
)
plt.axhline(0, color="black", linestyle="--", linewidth=0.7)
plt.xlabel("Time (Periods)")
plt.ylabel("Response")
plt.title("IRF of Inflation to One-Standard-Deviation MBC Shock")
plt.legend()
plt.grid()

plt.subplot(4, 1, 2)
plt.plot(
    irfs_q["unemployment_rate"],
    label="IRF: U to 1-Std MBC Shock",
)
plt.axhline(0, color="black", linestyle="--", linewidth=0.7)
plt.xlabel("Time (Periods)")
plt.ylabel("Response")
plt.title("IRF of Unemployment to One-Standard-Deviation MBC Shock")
plt.legend()
plt.grid()

plt.subplot(4, 1, 4)
plt.plot(irfs["gamma_f"], label=r"$\pi_{t+12}^{1y} - \pi_{t-1}^{1y}$")
plt.axhline(0, color="red", linestyle="--", linewidth=0.8)
plt.title(r"$\pi_{t+12}^y - \pi_{t-1}^y$")
plt.xlabel("Time")
plt.ylabel("Difference")
plt.legend()
plt.grid()

plt.subplot(4, 1, 3)
plt.plot(irfs["y"], label=r"$\pi_{t}^{1m} - \pi_{t-1}^{1y}$")
plt.axhline(0, color="red", linestyle="--", linewidth=0.8)
plt.title(r"$\pi_{t}^{1m} - \pi_{t-1}^{1y}$")
plt.xlabel("Time")
plt.ylabel("Difference")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

reg = smf.ols(
    "y ~ gamma_f + unemployment_rate - 1",
    data=irfs_q.iloc[:12],
).fit()
print(reg.summary())

# Matrix
ZM = np.linalg.inv(sqrtm((Z @ M_X @ Z.T) / T)) @ Z @ M_X
Theta_Y = np.hstack(
    [
        irfs["gamma_f"].iloc[horizons].to_numpy()[:H],
        irfs["unemployment_rate"].iloc[horizons].to_numpy()[:H],
    ]
).reshape(N_Y * H, 1)
Theta_y = irfs["y"].iloc[horizons].to_numpy()[:H].reshape(H, N_z)
YP = Theta_Y @ ZM

# First Stage Errors
Q = (Z @ Z.T) / T
v_H_perp = Y_H_perp - np.sqrt(1 / Q) * Theta_Y @ Z.reshape(1, T)

# KLM


def u1(b: np.ndarray) -> np.ndarray:
    """_summary_

    Args:
        b (np.ndarray): Loadings matrix of shape N_Y x 1

    Returns:
        np.ndarray: residual of the second stage SP-IV of shape H x T
    """
    return y_H_perp - np.kron(b.T, np.eye(H)) @ Y_H_perp


def u2(b: np.ndarray) -> np.ndarray:
    """_summary_

    Args:
        b (np.ndarray): Loadings matrix of shape N_Y x 1

    Returns:
        np.ndarray: residual of the IRF regression SP-IV of shape H x 1
    """
    return (Theta_y - np.kron(b.T, np.eye(H)) @ Theta_Y) @ ZM


def KLM(b: np.ndarray) -> np.ndarray:
    Sigma_inv = np.linalg.inv((u1(b) - u2(b)) @ u1(b).T)
    Y_tilde = YP - v_H_perp @ (u1(b) - u2(b)).T @ np.linalg.inv(
        (u1(b) - u2(b)) @ (u1(b) - u2(b)).T
    ) @ u2(b)
    term1 = (Sigma_inv @ u1(b) @ Y_tilde.T).flatten().T @ R.T
    term2 = (
        R @ np.kron(Y_tilde @ Y_tilde.T, Sigma_inv @ u1(b) @ u1(b).T @ Sigma_inv) @ R.T
    )
    term3 = R @ (Sigma_inv @ u1(b) @ Y_tilde.T).flatten().T
    return (T - N_X - N_z) * term1 @ np.linalg.inv(term2) @ term3


estimated_coef = reg.params

critical_values = {
    "68%": chi2.ppf(0.68, df=2),
    "90%": chi2.ppf(0.90, df=2),
    "95%": chi2.ppf(0.95, df=2),
}


def compute_klm_grid(gamma_f_grid, lambda_grid):
    """
    Compute KLM statistics over a parameter grid.

    Returns:
    - klm_grid: Grid of KLM statistics for each combination of γ_f and λ.
    """
    klm_grid = np.zeros_like(gamma_f_grid)
    for i in range(gamma_f_grid.shape[0]):
        for j in range(gamma_f_grid.shape[1]):
            simulated_params = np.array([gamma_f_grid[i, j], lambda_grid[i, j]])
            klm_grid[i, j] = KLM(simulated_params)
    return klm_grid  # instead compute what depends on the parameter in KLM


def plot_confidence_regions(gamma_f_grid, lambda_grid, coef):
    """
    Plot the confidence regions for the KLM statistic.
    """
    klm_grid = compute_klm_grid(gamma_f_grid, lambda_grid)
    plt.figure(figsize=(8, 6))
    colors = ["red", "orange", "green"]  # Add more colors if needed
    for (level, crit_val), color in zip(critical_values.items(), colors):
        plt.contour(
            gamma_f_grid,
            lambda_grid,
            klm_grid,
            levels=[crit_val],
            linewidths=1.5,
            colors=color,
            label=f"{level} CI",
        )

    # Add point estimates
    plt.scatter([coef[0]], [coef[1]], color="black", label="Point Estimate")

    plt.xlabel(r"$\gamma_f$")
    plt.ylabel(r"$\lambda$")
    plt.title("Confidence Sets Based on KLM Statistic")
    plt.legend()
    plt.grid()
    plt.show()


gamma_f_range = np.linspace(-0.8, 2, 100)
lambda_range = np.linspace(1, 3, 100)
gamma_f_grid, lambda_grid = np.meshgrid(gamma_f_range, lambda_range)

plot_confidence_regions(gamma_f_grid, lambda_grid, estimated_coef)
