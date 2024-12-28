# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 15:39:12 2024

@author: Maxime Coulet, Nina Stizi, Eliott Von-Pine

Prepare data for VAR estimation 
"""

def clean_data(data):
    """Compute spectral density matrix for the VAR model at a given frequency."""
    
    # Construct variables
    # Shift variables as required
    data['pi_1y_t_minus_1'] = data['core_CPI_yearly_percent_change'].shift(1)  # pi_1y at t-1
    data['pi_1y_t_plus_12'] = data['core_CPI_yearly_percent_change'].shift(-12)  # pi_1y at t+12
    
    # Calculate the dependent variable:
    data['pi_dep'] = data['core_CPI_annualised_percent_change'] - data['pi_1y_t_minus_1']
    
    # Calculate the independent variables
    data['pi_1y_diff'] = data['pi_1y_t_plus_12'] - data['pi_1y_t_minus_1']  
    
    var_data = data.iloc[780:1286,] # keep Jan 1978 to Feb 2020
    MBCshock_data = var_data
    
    var_columns = ['pi_dep','core_CPI_annualised_percent_change', 'unemployment_rate', 
                '10Y_Treasury_rate','3M_Treasury_rate',
                'log_industrial_production', 'PPI_commodity']
    
    var_data = var_data[var_columns]

    mbc_columns = ['core_CPI_annualised_percent_change', 'unemployment_rate', 
                   'log_industrial_production', 'PPI_commodity',
                '10Y_Treasury_rate','3M_Treasury_rate']
    
    MBCshock_data = var_data[mbc_columns]


    return var_data, MBCshock_data


