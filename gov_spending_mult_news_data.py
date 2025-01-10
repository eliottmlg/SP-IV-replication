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

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

from datetime import datetime
from statsmodels.tsa.api import VAR
from scipy.linalg import sqrtm, cholesky
from scipy.stats import chi2
from scipy.optimize import root_scalar, root, minimize_scalar

## Preprocessing

# Load the data
govdat = pd.read_csv("data/government_spending/govdat3908.csv")

# Remove data before start_time
start_date = datetime(
    1947, 1, 1
)  # or 1939 to include WWII but we end up with infinitely large confidence set
govdat = govdat[govdat["quarter"] >= start_date.year]
num_obs = len(govdat)

# Create quarter date index
govdat["qdate"] = pd.date_range(start=start_date, periods=num_obs, freq="QE")
govdat.set_index("qdate", inplace=True)

# Set dtypes
govdat.replace(".", np.nan, inplace=True)
govdat = govdat.astype(float)

# Create linear and quadratic trend
govdat["t"] = govdat["quarter"]
govdat["t2"] = govdat["quarter"] ** 2

# Create rwbus variable
govdat["rwbus"] = govdat["nwbus"] / govdat["pbus"]
govdat["rwmfg"] = govdat["nwmfg"] / govdat["pman"]
govdat["pdvmily"] = govdat["pdvmil"] / govdat["ngdp"].shift(1).fillna(233.8)

# Create log variables for the variables in the list
varlist = [
    "rgdp",
    "rcons",
    "rcnd",
    "rcsv",
    "rcdur",
    "rcndsv",
    "rinv",
    "rinvfx",
    "rnri",
    "rres",
    "tothours",
    "tothoursces",
    "rgov",
    "rdef",
]

for var in varlist:
    govdat[f"l{var}"] = np.log(govdat[var] / govdat["totpop"])

# Log variables for other variables in the list
varlist2 = ["totpop", "rwbus", "cpi", "pgdp"]
for var in varlist2:
    govdat[f"l{var}"] = np.log(govdat[var])

# Average Ratio GDP to Gov Spending from 1947 to 2008
avg_ratio = govdat["ngdp"].sum() / govdat["ngov"].sum()

## Build the VAR model using the standard identification method

# Select the columns for the model
vars_for_var = [
    "pdvmily",
    "lrgov",
    "lrgdp",
    "tb3",
    "amtbr",
    "ltothours",
]

# Param
horizons = range(12)
H = len(horizons)
num_var = len(vars_for_var)
burn_in = 0
lag_order = 4
num_ins = 1
num_reg = 1
T = num_obs - lag_order - len(horizons) - burn_in - 1

# Regression
target = "lrgdp"
regressor = "lrgov"
instruments = "pdvmily"

# Create the VAR model
model = VAR(govdat[vars_for_var], exog=govdat[["t", "t2"]])

# Fit the model
var_result = model.fit(lag_order, trend="n")

# Get the impulse response function (IRF)
irf = var_result.irf(max(horizons))

# Plot the IRF for government spending shock
irf.plot(impulse=instruments, signif=0.68, orth=True, figsize=(6, 20))
plt.show()

# Matrix
X = govdat[vars_for_var].to_numpy()
exog = govdat[["t", "t2"]].to_numpy()
X_stacked = np.hstack(
    [X[burn_in + lag : burn_in + T + lag] for lag in range(lag_order)]
).T

MX = np.eye(T) - X_stacked.T @ np.linalg.inv(X_stacked @ X_stacked.T) @ X_stacked

Z = X[burn_in + lag_order : burn_in + T + lag_order, vars_for_var.index(instruments)].T

R = np.kron(np.eye(num_reg), np.eye(H).flatten())

X_H = np.hstack(
    [X[lag_order + burn_in + h + 1 : lag_order + T + burn_in + h + 1] for h in horizons]
).T
y_H = np.vstack([X_H[h * num_var + vars_for_var.index(target)] for h in horizons])
Y_H = np.vstack([X_H[h * num_var + vars_for_var.index(regressor)] for h in horizons])

# Forecast Matrix
X_H_fcst = np.zeros(X_H.shape)
for t in range(T):
    X_t = X[: burn_in + t + lag_order]
    for h in horizons:
        fcst = var_result.forecast(
            y=X_t,
            exog_future=exog[
                burn_in + t + lag_order + 1 : burn_in + t + lag_order + 1 + h + 1
            ],
            steps=h + 1,
        )[-1]
        X_H_fcst[h * num_var : (h + 1) * num_var, t] = (
            fcst  # try to not use var instead estimate yourself with the stacked representation, just a simple OLS even with an exogenous variable just add in X_t
        )

# Error Forecast Matrix
X_H_perp = X_H - X_H_fcst
y_H_perp = np.vstack(
    [X_H_perp[h * num_var + vars_for_var.index(target)] for h in horizons]
)
Y_H_perp = np.vstack(
    [X_H_perp[h * num_var + vars_for_var.index(regressor)] for h in horizons]
)

# IRFs
IRFs = irf.orth_irfs
Gov_Spending_IRF = irf.orth_irfs[
    :, vars_for_var.index(regressor), vars_for_var.index(instruments)
]
GDP_IRF = irf.orth_irfs[:, vars_for_var.index(target), vars_for_var.index(instruments)]

# Government Spending Shock
cov_matrix = var_result.sigma_u
resid = var_result.resid.T
P = cholesky(cov_matrix, lower=True)
Shocks = (np.linalg.inv(P) @ resid).to_numpy()
War_Date_Shock = Shocks[vars_for_var.index(instruments)]
Z_perp = War_Date_Shock[burn_in + lag_order + 1 : burn_in + T + lag_order + 1].T

# First Stage error SP-IV
Q = (Z_perp @ Z_perp.T) / T
v_H_perp = Y_H_perp - np.sqrt(1 / Q) * Gov_Spending_IRF.reshape(
    H, num_ins
) @ Z_perp.reshape(num_ins, T)

# Second Stage SP-IV
reg = sm.OLS(GDP_IRF, Gov_Spending_IRF, hasconst=False).fit()
print(reg.summary())  # Elsaticity of GDP w.r.t Gov Spending
Gov_Spending_Mult = reg.params[0] * avg_ratio  # Gov Spending Multiplier

# Second Stage error SP-IV
ZM = np.sqrt(T / (Z @ MX @ Z.T)) * Z @ MX


def u1(b: np.ndarray) -> np.ndarray:
    return y_H_perp - np.kron(b.T, np.eye(H)) @ Y_H_perp


def u2(b: np.ndarray) -> np.ndarray:
    return (GDP_IRF - np.kron(b.T, np.eye(H)) @ Gov_Spending_IRF).reshape(
        H, num_ins
    ) @ ZM.reshape(num_ins, T)


# KLM
YP = Gov_Spending_IRF.reshape(H, num_ins) @ ZM.reshape(num_ins, T)


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
    return (T - num_var * 4 - num_ins) * term1 @ np.linalg.inv(term2) @ term3


def SP_IV_summary(
    coef, klm_stat, conf_levels=[0.95, 0.90, 0.68], scale=1, epsilon=10e-4
):
    """
    Generate a summary for an estimated coefficient using KLM statistics.

    Parameters:
    coef (float): Estimated coefficient (theta_hat).
    klm_stat (float): KLM statistic for the coefficient.
    fisher_info (float): Fisher information (inverse of variance).
    conf_levels (list): List of confidence levels (e.g., [0.95, 0.90, 0.68]).

    Returns:
    str: Summary of the coefficient estimation.
    """
    # Calculate p-value associated with the KLM statistic (chi2 with 1 dof)
    p_value = 1 - chi2.cdf(klm_stat, df=1)

    # Confidence intervals
    conf_intervals = {}
    for level in conf_levels:
        # Critical value for chi2
        c = chi2.ppf(level, df=1)

        # Solve KLM(theta) = c
        def error_function(b):
            return abs(KLM(np.array([b])) - c)

        lower = minimize_scalar(error_function, bounds=(-1, coef), method="bounded")
        lower_bound = lower.x if lower.fun < epsilon else -np.inf
        upper = minimize_scalar(error_function, bounds=(coef, 1), method="bounded")
        upper_bound = upper.x if upper.fun < epsilon else np.inf
        conf_intervals[level] = (lower_bound, upper_bound)

    # Formatting the summary
    summary = f"{'Summary of Coefficient Estimation':^50}\n"
    summary += "=" * 50 + "\n"
    summary += f"Estimated Coefficient: {scale*coef:.4f}\n"
    summary += f"KLM Statistic: {klm_stat:.4f}\n"
    summary += "-" * 50 + "\n"
    summary += f"{'Confidence Intervals':<20} {'Lower':>15} {'Upper':>15}\n"
    summary += "-" * 50 + "\n"

    for level, (lower, upper) in conf_intervals.items():
        summary += f"{level*100:>5.1f}% Confidence Interval {scale*lower:>15.4f} {scale*upper:>15.4f}\n"

    return summary


# SP-IV summary
coef = reg.params[0]  # Estimated coefficient (theta_hat)
klm_stat = KLM(
    np.array([coef])
)  # KLM statistic for the coefficient  # Fisher information

print(SP_IV_summary(coef, klm_stat, scale=avg_ratio))
