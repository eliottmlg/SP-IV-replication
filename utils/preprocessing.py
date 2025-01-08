# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 15:39:12 2024

@author: Maxime Coulet, Nina Stizi, Eliott Von-Pine

Prepare data for VAR estimation 
"""

import numpy as np
import pandas as pd


def clean_data(data):
    """Prepare data for var estimation."""

    # 12 month change of log IP
    data["log_industrial_production"] = data["log_industrial_production"] - data[
        "log_industrial_production"
    ].shift(12)

    var_data = data.iloc[780:1286,]  # keep Jan 1978 to Feb 2020
    MBCshock_data = var_data

    var_columns = [
        "core_CPI_annualised_percent_change",
        "unemployment_rate",
        "10Y_Treasury_rate",
        "3M_Treasury_rate",
        "log_industrial_production",
        "PPI_commodity",
    ]

    var_data = var_data[var_columns]

    mbc_columns = [
        "core_CPI_annualised_percent_change",
        "unemployment_rate",
        "log_industrial_production",
        "PPI_commodity",
        "10Y_Treasury_rate",
        "3M_Treasury_rate",
    ]

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
    cumulative_12_month_inflation = (1 + monthly_inflation).rolling(window=12).apply(
        np.prod, raw=True
    ) - 1

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
    cumulative_3_month_inflation = (1 + monthly_inflation).rolling(window=3).apply(
        np.prod, raw=True
    ) - 1

    # Annualize the 3-month inflation
    annualized_inflation = (1 + cumulative_3_month_inflation) ** (12 / 3) - 1

    return annualized_inflation


def compute_year_on_year(window, scale: float = 1):
    compounded_change = 1
    for val in window:
        compounded_change *= (1 + val / scale) ** (
            1 / 12
        )  # Apply the monthly annualized change directly as it's already in decimal form

    # Return the compounded year-on-year change (percentage)
    return (compounded_change - 1) * scale
