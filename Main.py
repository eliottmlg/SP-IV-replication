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


### Constructing MBC shock

# Load the data
file_path = 'data/monthly.csv'
data = pd.read_csv(file_path)

# Prepare data for VAR estimation 
var_data,MBCshock_data = prep.clean_data(data)
var_data.to_csv(r'data/var_data.csv',index=False)
MBCshock_data.to_csv(r'data/mbc_data.csv',index=False,header=False)
 
### Data formatting for SP-IV

# Path to your .mat file
file_path = 'data/MBC_IRFs.csv'
MBC_irfs = pd.read_csv(file_path,header=0)

# constructing IRFs for annual inflation based on IRFs for monthly annualised inflation
# 
theta_y = MBC_irfs
theta_Y = data



### SP-IV
