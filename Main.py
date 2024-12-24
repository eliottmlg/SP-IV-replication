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
var_data = prep.clean_data(data)

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
mbc_shock = mbc.constuctMBC_bandpass(residuals)

# method 3
mbc_shock,mbc_contributions = mbc.constuctMBC(var_data, fitted_model)
print("MBC Shock (first singular vector):", mbc_shock)
print("Contribution to variance:", mbc_contributions)


### SP-IV
