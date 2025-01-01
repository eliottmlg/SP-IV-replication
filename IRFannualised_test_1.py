# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 22:00:35 2025

@author: eliot
"""

# Function to compute pi_t^{1y} from pi_t^{1m}
def compute_year_over_year(pi_1m, window=12):
    # De-annualize monthly rates (convert to simple monthly growth rates)
    monthly_growth = pi_1m / 12  # Convert percent to decimal and de-annualize
    T = len(monthly_growth)
    pi_1y = []

    # Compute year-over-year percent change
    for t in range(window - 1, T):  # Start from the 12th month
        cumulative_growth = np.prod(1 + monthly_growth[t - window + 1:t + 1])  # Cumulative product
        year_over_year = (cumulative_growth - 1) * 100  # Convert to percent
        pi_1y.append(year_over_year)
    
    return np.array(pi_1y)


# Compute pi_t^{1y}
pi_1y = compute_year_over_year(theta_y)
pi_1y = pd.DataFrame(difference_irf)
pi_1y.plot() 
plt.show()



# Compute pi_{t+12}^y and pi_{t-1}^y
def compute_difference(pi_1y, shift_forward=12, shift_backward=1):
    # Shift forward for pi_{t+12}^y
    pi_t_plus_12_y = np.roll(pi_1y, -shift_forward)
    pi_t_plus_12_y[-shift_forward:] = np.nan  # Fill last values with NaN (incomplete data)
    
    # Shift backward for pi_{t-1}^y
    pi_t_minus_1_y = np.roll(pi_1y, shift_backward)
    pi_t_minus_1_y[:shift_backward] = np.nan  # Fill first values with NaN (incomplete data)
    
    # Compute the difference
    difference = pi_t_plus_12_y - pi_t_minus_1_y
    
    return difference

# Compute pi_{t+12}^y - pi_{t-1}^y
difference_irf = compute_difference(pi_1y)
difference_irf = pd.DataFrame(difference_irf)
difference_irf.plot() 
plt.show()

theta_Y = pd.concat([difference_irf, X_u], axis=1) 
theta_Y = theta_Y.iloc[0:H].dropna().reset_index(drop=True) # cut at chosen horizon
theta_y = theta_y.iloc[0:H].iloc[0:len(theta_Y)]

# Fit the regression model
model = sm.OLS(theta_y, theta_Y).fit()

# Print a detailed summary
print(model.summary())
