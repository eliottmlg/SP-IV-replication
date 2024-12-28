import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from scipy.linalg import svd
import matplotlib.pyplot as plt

data = var_data
var_results = fitted_model


# Step 2: Define the spectral density function for variable k
def compute_spectral_density_for_k(var_results, freq, k_index):
    """
    Compute the spectral density for a specific variable k at a given frequency.
    
    Parameters:
    - var_results: Fitted VAR model results
    - freq: Frequency to compute the spectral density
    - k_index: Index of the target variable (0-based index)

    Returns:
    - Spectral density matrix for variable k
    """
    omega = 2 * np.pi * freq  # Angular frequency
    I = np.eye(var_results.neqs)  # Identity matrix
    A = np.sum(
        [var_results.coefs[i] * np.exp(-1j * omega * (i + 1)) for i in range(var_results.k_ar)],
        axis=0,
    )  # Fourier-transform VAR coefficients
    C_k = np.linalg.inv(I - A)  # Transfer function
    return C_k[k_index, :]  # Extract row corresponding to variable k

# Step 3: Integrate the spectral density over the frequency band
low_freq = 1 / 32
high_freq = 1 / 6
freqs = np.linspace(low_freq, high_freq, 500)

# Choose the index of the target variable (e.g., unemployment)
k_index = data.columns.get_loc('unemployment_rate')  # Replace 'unemployment' with your variable

spectral_density_sum_k = np.zeros((data.shape[1], data.shape[1]), dtype=complex)
for freq in freqs:
    C_k = compute_spectral_density_for_k(var_results, freq, k_index)
    spectral_density_sum_k += np.outer(C_k.conj(), C_k)  # Outer product for variable k

spectral_density_sum_k /= len(freqs)  # Normalize

# Step 4: Perform SVD to find the dominant shock direction for variable k
spectral_density_real_k = np.real(spectral_density_sum_k)
U_k, S_k, Vt_k = svd(spectral_density_real_k)
q_k = U_k[:, 0]  # First singular vector corresponds to the dominant shock

# Step 5: Compute IRFs for the identified shock
irf = var_results.irf(steps=20)  # Compute IRFs for 20 periods
shock_adjusted_irfs = irf.orth_irfs @ q_k  # Adjust IRFs for the identified shock

# Plot IRFs
variables = data.columns  # Variable names
for i, var in enumerate(variables):
    plt.figure()
    plt.plot(shock_adjusted_irfs[:, i], label=f"IRF of {var}")
    plt.axhline(0, color='black', linestyle='--', linewidth=0.7)
    plt.title(f"IRF of {var} to the Identified Shock")
    plt.xlabel("Periods")
    plt.ylabel("Response")
    plt.legend()
    plt.show()

# Step 6: Compute FEVD for the identified shock
variance_contribution = {}
for i, var in enumerate(variables):
    fevd_mbc = []  # Store variance contributions over forecast horizons
    for h in range(20):  # Forecast horizon (20 steps)
        # Total variance of variable i
        total_variance = np.sum(np.square(irf.orth_irfs[:h + 1, i, :]))

        # Variance contribution of the identified shock
        mbc_variance = np.sum(np.square(shock_adjusted_irfs[:h + 1, i]))

        # Fractional contribution
        fractional_contribution = mbc_variance / total_variance if total_variance > 0 else 0
        fevd_mbc.append(fractional_contribution)

    # Store FEVD for the variable
    variance_contribution[var] = fevd_mbc

# Plot FEVD
for var, fevd in variance_contribution.items():
    plt.figure()
    plt.plot(range(1, 21), fevd, label=f"FEVD: {var}")
    plt.xlabel("Horizon (Periods)")
    plt.ylabel("Fractional Contribution")
    plt.title(f"FEVD of {var} to the Identified Shock")
    plt.axhline(0, color='black', linestyle='--', linewidth=0.7)
    plt.legend()
    plt.show()
