
import numpy as np
import scipy.linalg as la
import statsmodels.api as sm
from scipy.signal import periodogram
from statsmodels.tsa.api import VAR

# Configurations based on Julia code
gbseed = 1234567890
np.random.seed(gbseed)

alpha = 68
lags = 4

# Frequency domain settings for Business Cycle Analysis
wmin = 2 * np.pi / 32
wmax = 2 * np.pi / 6

# Load data
data = var_data

# Estimate standard VAR model
model = VAR(data)
res = model.fit(lags)

# Estimate standard VAR model
model = VAR(data)
res = model.fit(lags)

#data.columns.get_loc('unemployment_rate')

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
    def frequency_response(matrix, freqs):
        responses = []
        for freq in freqs:
            z = np.exp(-1j * freq)  # Complex exponential for frequency response
            response = np.linalg.inv(np.eye(matrix.shape[0]) - np.sum([matrix[l] * z**(l + 1) for l in range(len(matrix))], axis=0))
            responses.append(response)
        return np.array(responses)

    freqs = np.linspace(wmin, wmax, 100)
    freq_responses = frequency_response(var_res.coefs, freqs)

    # Select target variable's spectral density
    spectral_density = np.sum(np.abs(freq_responses[:, :, target_variable])**2, axis=0)
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
