# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 22:00:35 2025

@author: eliot
"""
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

def compute_difference_t_minus_1(pi_1m, pi_1y):
    # Compute pi_t^m - pi_{t-1}^y
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
theta_Y = theta_Y.iloc[0:H].dropna().reset_index(drop=True) # cut at chosen horizon
pi_t_minus_1 = pi_t_minus_1.dropna().reset_index(drop=True)
pi_t_minus_1 = pi_t_minus_1.iloc[0:H].iloc[0:len(theta_Y)]

# Fit the regression model
model = sm.OLS(pi_t_minus_1, theta_Y).fit()

# Print a detailed summary
print(model.summary())
