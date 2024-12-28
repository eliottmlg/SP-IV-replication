import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from scipy.signal import welch
from scipy.linalg import svd

# Generate or load your macroeconomic data
data = var_data
var_results = fitted_model

var_results.resid

# Set the frequency range for business cycles (6-32 quarters)
low_freq = 1 / 32
high_freq = 1 / 6

# Step 2: Compute the spectral density matrix
def compute_spectral_density(var_results, freq):
    """Compute spectral density matrix for a given frequency."""
    omega = 2 * np.pi * freq
    I = np.eye(var_results.neqs)
    A = np.sum([var_results.coefs[i] * np.exp(-1j * omega * (i + 1))
                for i in range(var_results.k_ar)], axis=0)
    return np.linalg.inv(I - A) @ var_results.sigma_u @ np.linalg.inv(I - A).conj().T

# Step 3: Integrate variance contributions over the 6-32 quarters frequency band
freqs = np.linspace(low_freq, high_freq, 500)
spectral_density_sum = np.zeros((data.shape[1], data.shape[1]), dtype=complex)

for freq in freqs:
    spectral_density_sum += compute_spectral_density(var_results, freq)

# Normalize by the frequency band width
spectral_density_sum /= len(freqs)

# Ensure spectral_density_sum is real for SVD
spectral_density_sum_real = np.real(spectral_density_sum)

# Step 4: Perform singular value decomposition (SVD) on the integrated spectral density matrix
U, S, Vt = svd(spectral_density_sum_real)

# Find the vector that maximizes unemployment fluctuations
unemployment_index = data.columns.get_loc('unemployment_rate')
max_contribution_vector = U[:, unemployment_index]

# Step 5: Compute the time series for the shock maximizing unemployment fluctuations
shock_series = var_results.resid @ max_contribution_vector

# Step 6: Output results
print("Shock maximizing cyclical unemployment fluctuations:", max_contribution_vector)
print("Time series for the shock:", shock_series)

# Save the series for later use
shock_series_df = pd.DataFrame({"Unemployment_Shock": shock_series}, index=data.index)
shock_series_df.to_csv("unemployment_shock_series.csv")
