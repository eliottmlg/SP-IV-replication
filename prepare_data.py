# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 15:39:12 2024

@author: Maxime Coulet, Nina Stizi, Eliott Von-Pine

Prepare data for VAR estimation 
"""

import numpy as np
import pandas as pd

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


def compute_annual_inflation_from_monthly_annualized(monthly_annualized_inflation):
    """
    Compute annual inflation rates from a time series of monthly annualized inflation rates.

    Parameters:
    monthly_annualized_inflation (pandas Series): Time series of monthly annualized inflation rates (as decimals).

    Returns:
    pandas Series: Time series of annual inflation rates.
    """
  
    # Convert monthly annualized inflation to monthly inflation
    monthly_inflation = (1 + monthly_annualized_inflation) ** (1 / 12) - 1

    # Compute the rolling 12-month cumulative inflation
    cumulative_12_month_inflation = (1 + monthly_inflation).rolling(window=12).apply(np.prod, raw=True) - 1

    return cumulative_12_month_inflation


def from_monthly_annualized_to_Q(monthly_annualized_inflation):
    """
    Compute quarterly inflation rates from a time series of monthly annualized inflation rates.

    Parameters:
    monthly_annualized_inflation (pandas Series): Time series of monthly annualized inflation rates (as decimals).

    Returns:
    pandas Series: Time series of annual inflation rates.
    """
    # Convert monthly annualized inflation to monthly inflation
    monthly_inflation = (1 + monthly_annualized_inflation) ** (1 / 12) - 1

    # Compute 3-month cumulative inflation (one quarter)
    cumulative_3_month_inflation = (1 + monthly_inflation).rolling(window=3).apply(np.prod, raw=True) - 1

    # Annualize the 3-month inflation
    annualized_inflation = (1 + cumulative_3_month_inflation) ** (12 / 3) - 1

    
    return annualized_inflation