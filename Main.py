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
# run these 2 below to save
#var_data.to_csv(r'data/var_data.csv',index=False)
#MBCshock_data.to_csv(r'data/mbc_data.csv',index=False,header=False)

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

MBCshock_timeseries = np.dot(var_residuals, MBCshock) # MBC shock time series

## construct IRFs to mbc shock
std_mbc_shock = np.sqrt(np.dot(np.dot(MBCshock.T, sigma_u), MBCshock))  # Standard deviation of MBC
MBCshock_normalized = MBCshock / std_mbc_shock  # Normalize the shock vector
#MBCshock_normalized = MBCshock / 1  # don't Normalize the shock vector

forecast_steps = 120
irf = var_results.irf(periods=forecast_steps)  # IRF object
irfs = irf.orth_irfs  # Shape: (forecast_steps, num_variables, num_variables)

# Adjust IRFs for the one-standard-deviation MBC shock
shock_adjusted_irfs = irfs @ MBCshock_normalized.to_numpy() # Weighted IRFs for the 1-std MBC shock

irf_inflation_to_mbc = shock_adjusted_irfs[:,0][0:-1:3]
# unemployment
X_u = shock_adjusted_irfs[:,1][0:-1:3] # take only the 0th, 3th, 6th, ... observations

# Plot the IRF
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(range(20), irf_inflation_to_mbc[0:20], label="IRF: $\pi^m$ to 1-Std MBC Shock")
plt.axhline(0, color='black', linestyle='--', linewidth=0.7)
plt.xlabel("Time (Periods)")
plt.ylabel("Response")
plt.title("IRF of Inflation to One-Standard-Deviation MBC Shock")
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(range(20), X_u[0:20], label="IRF: U to 1-Std MBC Shock")
plt.axhline(0, color='black', linestyle='--', linewidth=0.7)
plt.xlabel("Time (Periods)")
plt.ylabel("Response")
plt.title("IRF of Unemployment to One-Standard-Deviation MBC Shock")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
 
### Data formatting for SP-IV
H = 12 # chosen horizon to keep in IRFs

shock_adjusted_irfs = shock_adjusted_irfs.reshape(121,6)
shock_adjusted_irfs = pd.DataFrame(shock_adjusted_irfs)

# unemployment
X_u = shock_adjusted_irfs[[1]]
X_u = X_u[0:-1:3] # take only the 0th, 3th, 6th, ... observations
X_u = X_u.reset_index(drop=True)
X_u = X_u.iloc[0:H]

# dependent inflation (LHS) and independent inflation (RHS)
pi_1m = shock_adjusted_irfs[[0]]

# Function to compute pi_t^{1y} from pi_t^{1m}
def compute_year_over_year(pi_1m, window=12):
    # De-annualize monthly rates (convert to simple monthly growth rates)
    monthly_growth = (1 + pi_1m) ** (1 / 12) - 1
    T = len(monthly_growth)
    pi_1y = []

    # Compute year-over-year percent change
    for t in range(window - 1, T):  # Start from the 12th month
        cumulative_growth = np.prod(1 + monthly_growth[t - window + 1:t + 1])  # Cumulative product
        year_over_year = (cumulative_growth - 1)  # Convert to percent
        pi_1y.append(year_over_year)
    
    return np.array(pi_1y)

# Compute pi_t^{1y}
pi_1y = compute_year_over_year(pi_1m)

# Compute pi_{t+12}^y - pi_{t-1}^y
def compute_difference_12_minus_1(pi_1y, shift_forward=12, shift_backward=1):
    # Shift forward for pi_{t+12}^y
    pi_t_plus_12_y = np.roll(pi_1y, -shift_forward)
    pi_t_plus_12_y[-shift_forward:] = np.nan  # Fill last values with NaN (incomplete data)
    
    # Shift backward for pi_{t-1}^y
    pi_t_minus_1_y = np.roll(pi_1y, shift_backward)
    pi_t_minus_1_y[:shift_backward] = np.nan  # Fill first values with NaN (incomplete data)
    
    # Compute the difference
    difference_12_minus_1 = pi_t_plus_12_y - pi_t_minus_1_y
    
    return difference_12_minus_1

# Compute pi_t^m - pi_{t-1}^y
def compute_difference_t_minus_1(pi_1m, pi_1y):
    # Shift backward for pi_{t-1}^y
    pi_t_minus_1_y = np.roll(pi_1y, 1)
    pi_t_minus_1_y[:1] = np.nan  # Fill first value with NaN (incomplete data)

    # Compute the difference
    difference_t_minus_1 = pi_1m[:len(pi_1y)] - pi_t_minus_1_y

    return difference_t_minus_1

# Compute differences
pi_12_minus_1 = compute_difference_12_minus_1(pi_1y)
pi_t_minus_1 = compute_difference_t_minus_1(pi_1m, pi_1y)

# take only 0th, 3th, 6th, ... horizons
pi_12_minus_1 = pi_12_minus_1[0:-1:3]
pi_t_minus_1 = pi_t_minus_1[0:-1:3]

# Plot results
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(pi_12_minus_1, label="$\pi_{t+12}^y - \pi_{t-1}^y$")
plt.axhline(0, color='red', linestyle='--', linewidth=0.8)
plt.title("$\pi_{t+12}^y - \pi_{t-1}^y$")
plt.xlabel("Time")
plt.ylabel("Difference")
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(pi_t_minus_1, label="$\pi_{t}^m - \pi_{t-1}^y$")
plt.axhline(0, color='red', linestyle='--', linewidth=0.8)
plt.title("$\pi_{t}^m - \pi_{t-1}^y$")
plt.xlabel("Time")
plt.ylabel("Difference")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

pi_12_minus_1 = pd.DataFrame(pi_12_minus_1)
pi_t_minus_1 = pd.DataFrame(pi_t_minus_1)

theta_Y = pd.concat([pi_12_minus_1, X_u], axis=1) 
theta_Y = theta_Y.iloc[0:H]
theta_Y = theta_Y.dropna().reset_index(drop=True) # cut at chosen horizon
pi_t_minus_1 = pi_t_minus_1.dropna().reset_index(drop=True) # drop NA
pi_t_minus_1 = pi_t_minus_1.iloc[0:H].iloc[0:len(theta_Y)]

### SP-IV
# Fit the regression model
model = sm.OLS(pi_t_minus_1, theta_Y).fit()
# Print a detailed summary
print(model.summary())



