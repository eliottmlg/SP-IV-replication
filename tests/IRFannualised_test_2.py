
# Function to compute compounded annual IRF
def compute_compounded_irf(irf_monthly, start_index, end_index):
    # De-annualize the monthly rates
    monthly_growth = irf_monthly[start_index:end_index] / 12
    # Compute the compounded yearly rate
    compounded_growth = np.prod(1 + monthly_growth) ** 12 - 1
    return compounded_growth

# Construct IRF for pi_{t+12}^y - pi_{t-1}^y
def construct_difference_irf_annualized(irf_monthly, horizon=12):
    T = len(irf_monthly)
    diff_irf = []

    for t in range(horizon, T - horizon):  # Ensure sufficient range for leading/lagging
        pi_t_plus_12_y = compute_compounded_irf(irf_monthly, t, t + horizon)
        pi_t_minus_1_y = compute_compounded_irf(irf_monthly, t - horizon, t)
        diff_irf.append(pi_t_plus_12_y - pi_t_minus_1_y)

    return np.array(diff_irf)

# Compute the IRF for pi_{t+12}^y - pi_{t-1}^y
difference_irf = construct_difference_irf_annualized(theta_y)
difference_irf = pd.DataFrame(difference_irf)
difference_irf.plot() 
plt.show()

theta_Y = pd.concat([difference_irf, X_u], axis=1) 
theta_Y = theta_Y.iloc[0:H] # cut at chosen horizon
theta_y = theta_y.iloc[0:H]

# Fit the regression model
model = sm.OLS(theta_y, theta_Y).fit()

# Print a detailed summary
print(model.summary())
