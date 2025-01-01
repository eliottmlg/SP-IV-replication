# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 16:32:41 2024

@author: Maxime Coulet, Nina Stizi, Eliott Von-Pine
"""

#==============================================================================
# Advanced Macroeconometrics - Final assignement
# 
# Dynamic Identification Using System Projections on Instrumental Variables 
# Working Paper No. 2204 – Dallas Fed
#==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
import prepare_data as prep


### Constructing MBC shock

# Load the data
file_path = 'data/monthly.csv'
data = pd.read_csv(file_path)

# Prepare data for VAR estimation 
var_data,MBCshock_data = prep.clean_data(data)
var_data.to_csv(r'data/var_data.csv',index=False)
MBCshock_data.to_csv(r'data/mbc_data.csv',index=False,header=False)

# load mbc shock as linear combination of orthogonal shocks from Business Cycle Anatomy
file_path = 'data/MBC_shock.csv'
MBCshock = pd.read_csv(file_path,header=None)
MBCshock = MBCshock.T

# Fit a VAR model
var_model = VAR(MBCshock_data)
lag_order = 6
var_results = var_model.fit(lag_order)
var_residuals = var_results.resid
sigma_u = var_results.sigma_u  # Covariance matrix of residuals

## construct mbc shock (optional for replication)
# Decompose the covariance matrix using Cholesky decomposition
P = np.linalg.cholesky(sigma_u)  # Lower triangular matrix

# Compute structural shocks
structural_shocks = np.linalg.inv(P) @ var_residuals.T  # Shape: (num_variables, num_timepoints)
structural_shocks = structural_shocks.T  # Transpose to align with time series format

MBCshock_timeseries = np.dot(var_residuals, MBCshock) # MBC shock time series

## construct IRFs to mbc shock
std_mbc_shock = np.sqrt((MBCshock.T @ sigma_u) @ MBCshock)  # Standard deviation of MBC
#MBCshock_normalized = MBCshock / std_mbc_shock  # Normalize the shock vector
MBCshock_normalized = MBCshock / 1  # Normalize the shock vector

forecast_steps = 120
irf = var_results.irf(periods=forecast_steps)  # IRF object
irfs = irf.orth_irfs  # Shape: (forecast_steps, num_variables, num_variables)

# Adjust IRFs for the one-standard-deviation MBC shock
shock_adjusted_irfs = irfs @ MBCshock_normalized.to_numpy() # Weighted IRFs for the 1-std MBC shock

irf_inflation_to_mbc = shock_adjusted_irfs[:, 0]

# Plot the IRF
plt.figure()
plt.plot(range(60), irf_inflation_to_mbc[0:60], label="IRF: Inflation to 1-Std MBC Shock")
plt.axhline(0, color='black', linestyle='--', linewidth=0.7)
plt.xlabel("Time (Periods)")
plt.ylabel("Response")
plt.title("IRF of Inflation to One-Standard-Deviation MBC Shock")
plt.legend()
plt.show()
 
### Data formatting for SP-IV
H = 12 # chosen horizon to keep in IRFs

# Path to your .mat file
file_path = 'data/aaaIRF.csv'
MBC_irfs = pd.read_csv(file_path,header=None)

shock_adjusted_irfs = shock_adjusted_irfs.reshape(121,6)
shock_adjusted_irfs = pd.DataFrame(shock_adjusted_irfs)

# constructing IRFs for annual inflation based on IRFs for monthly annualised inflation
theta_y = MBC_irfs[[0]]
theta_y = shock_adjusted_irfs[[0]]
theta_y[0:61].plot() 
plt.show()

theta_y = theta_y[0:-1:3] # take only the 0th, 3th, 6th, ... observations
theta_y = theta_y.reset_index(drop=True)

# Yearly
inflation_yearly = prep.compute_annual_inflation_from_monthly_annualized(theta_y)
inflation_yearly = inflation_yearly.iloc[13:-1].reset_index(drop=True)

inflation_yearly_lag1 = inflation_yearly.shift(1) # lagging irf for annual inflation
inflation_yearly_lead12 = inflation_yearly.shift(-12) # leading irf for annual inflation
X_inflation = inflation_yearly_lead12 - inflation_yearly_lag1
X_inflation = X_inflation.iloc[0:13]

# for unemployment
X_u = MBC_irfs[[1]]
X_u = shock_adjusted_irfs[[1]]
X_u = X_u[0:-1:3] # take only the 0th, 3th, 6th, ... observations
X_u = X_u.reset_index(drop=True)
X_u = X_u.iloc[0:13]

theta_Y = pd.concat([X_inflation, X_u], axis=1) 
theta_Y = theta_Y.rename(columns={0: 'pi_12L-pi_1l', 1: 'U'})
theta_Y = theta_Y.iloc[0:H] # cut at chosen horizon
 # handle NA
theta_Y = theta_Y.dropna()
theta_Y = theta_Y.fillna(0.07)
theta_Y = theta_Y.reset_index(drop=True)

theta_y = theta_y.iloc[0:H]
theta_y = theta_y.rename(columns={0: 'piM-piY_1l'})

# plot
theta_y.iloc[0:21].plot() 
plt.show()

X_inflation.iloc[0:21].plot() 
plt.show()

X_u.iloc[0:21].plot() 
plt.show()


### SP-IV

# Fit the regression model
model = sm.OLS(theta_y, theta_Y).fit()

# Print a detailed summary
print(model.summary())


