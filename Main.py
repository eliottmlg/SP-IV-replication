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
from statsmodels.tsa.api import VAR
import prepare_data as prep
import construct_MBC as mbc

### Constructing MBC shock

# Load the data
file_path = 'data/monthly.csv'
data = pd.read_csv(file_path)

# Prepare data for VAR estimation 
var_data,MBCshock_data = prep.clean_data(data)
var_data.to_csv(r'data/var_data.csv',index=False)
MBCshock_data.to_csv(r'data/mbc_data.csv',index=False,header=False)
    
# Fit the VAR model
model = VAR(var_data)
fitted_model = model.fit(maxlags=6)  # Adjust the maximum lag as needed

# Display the fitted model summary
print(fitted_model.summary())

# Get the impulse response functions (IRFs)
irf = fitted_model.irf(12)  # Impulse responses for 12 periods
irf.plot(orth=True,impulse='pi_dep')  # Orthogonalized impulse response

# Extract residuals from the VAR
residuals = fitted_model.resid

# Construct MBC shock (i don't know which one works)
# method 1
mbc_shock = mbc.constructMBC_spectral(residuals)

# method 2
mbc_shock2 = mbc.constuctMBC_bandpass(residuals)

plt.plot(mbc_shock, label='MBC 1', color='blue', linewidth=2)
plt.plot(mbc_shock2, label='MBC 2', color='red', linewidth=2)

# method 3 --- closest replication of BC anatomy paper
mbc_shock,mbc_contributions = mbc.constuctMBC(var_data, fitted_model)
print("MBC Shock (first singular vector):", mbc_shock)
print("Contribution to variance:", mbc_contributions)

plt.plot(mbc_shock, label='MBC 1', color='blue', linewidth=2)

### Checking MBC shock

# Use the fitted VAR model to compute IRFs
irf = var_results.irf(steps=20)  # Compute IRFs for 20 periods

# Adjust IRFs to align with your shock
shock_adjusted_irfs = irf.orth_irfs @ mbc_contributions

# Plot the IRFs
variables = var_data.columns  # Variable names
for i, var in enumerate(variables):
    plt.figure()
    plt.plot(shock_adjusted_irfs[:, i], label=f"IRF of {var}")
    plt.axhline(0, color='black', linestyle='--', linewidth=0.7)
    plt.title(f"IRF of {var} to the Shock")
    plt.xlabel("Periods")
    plt.ylabel("Response")
    plt.legend()
    plt.show()

# Initialize dictionary to store FEVD for each variable
variance_contribution = {}

# Adjust orthogonal impulse responses using the MBC shock vector
shock_adjusted_irfs = irf.orth_irfs @ mbc_contributions  # Adjusted IRFs for the MBC shock

# Loop over each variable to calculate FEVD
for i, var in enumerate(variables):
    fevd_mbc = []  # Store variance contributions over forecast horizons
    for h in range(20):  # Forecast horizon (20 steps)
        # Total variance of variable i
        total_variance = np.sum(np.square(irf.orth_irfs[:h + 1, i, :]))

        # Variance contribution of the MBC shock
        mbc_variance = np.sum(np.square(shock_adjusted_irfs[:h + 1, i]))

        # Fractional contribution
        fractional_contribution = mbc_variance / total_variance if total_variance > 0 else 0
        fevd_mbc.append(fractional_contribution)

    # Store FEVD for the variable
    variance_contribution[var] = fevd_mbc

# Plot FEVD for each variable
for var, fevd in variance_contribution.items():
    plt.figure()
    plt.plot(range(1, 21), fevd, label=f"FEVD: {var}")
    plt.xlabel("Horizon (Periods)")
    plt.ylabel("Fractional Contribution")
    plt.title(f"FEVD of {var} to the MBC Shock")
    plt.axhline(0, color='black', linestyle='--', linewidth=0.7)
    plt.legend()
    plt.show()


### Data formatting for SP-IV

### SP-IV
