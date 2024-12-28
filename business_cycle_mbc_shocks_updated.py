
import numpy as np
import scipy.linalg as la
import statsmodels.api as sm
from scipy.signal import periodogram
from statsmodels.tsa.api import VAR

# Configurations based on Julia code
gbseed = 1234567890
np.random.seed(gbseed)

alpha = 68
lags = 6
start_year = 1955
end_year = 2017.75

# Frequency domain settings for Business Cycle Analysis
wmin = 2 * np.pi / 32
wmax = 2 * np.pi / 6

data = var_data

# Estimate standard VAR model
model = VAR(data)
res = model.fit(lags)

# Function to compute Cholesky decomposition for IRFs
def compute_irfs(var_res, n_steps):
    irfs = var_res.irf(n_steps).irfs
    return irfs

# Compute IRFs
n_steps = 40
irfs = compute_irfs(res, n_steps)

# Function to compute MBC shock targeting a specific variable
def compute_mbc_shock(var_res, target_variable, wmin, wmax):
    # Cholesky decomposition for initial identification
    chol_decomp = np.linalg.cholesky(var_res.sigma_u)

    # Frequency domain representation
    def frequency_response(coefficients, freqs):
        responses = []
        num_vars = coefficients.shape[1]
        for freq in freqs:
            z = np.exp(-1j * freq)  # Complex exponential for frequency response
            identity = np.eye(num_vars)
            summation = identity.copy()
            for lag, coef in enumerate(coefficients):
                summation -= coef @ (z ** (lag + 1))
            response = np.linalg.inv(summation)
            responses.append(response)
        return np.array(responses)

    freqs = np.linspace(wmin, wmax, 100)
    freq_responses = frequency_response(var_res.coefs, freqs)

    # Select target variable's spectral density
    spectral_density = np.sum(np.abs(freq_responses[:, target_variable, :])**2, axis=0)
    eigvals, eigvecs = np.linalg.eig(spectral_density)
    max_eigen_index = np.argmax(eigvals)
    mbc_shock = eigvecs[:, max_eigen_index].real

    return mbc_shock

# Compute MBC shocks for each variable
mbc_shocks = []
for target_variable in range(data.shape[1]):
    mbc_shock = compute_mbc_shock(res, target_variable, wmin, wmax)
    mbc_shocks.append(mbc_shock)

# Variance decomposition
vd = res.fevd(10).decomp

# Results storage (matching Julia code structure)
irf_results = np.mean(irfs, axis=0)  # Mean IRFs
var_decomp = np.mean(vd, axis=0)    # Mean variance decomposition

# Save results to file
np.savez("results.npz", irfs=irf_results, vd=var_decomp, mbc_shocks=mbc_shocks)
