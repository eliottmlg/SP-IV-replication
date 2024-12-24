# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 15:13:50 2024

@author: Maxime Coulet, Nina Stizi, Eliott Von-Pine

Constructing MBC shock
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from scipy.signal import welch
from scipy.linalg import svd

def constuctMBC(data, var_results):
    """Construct MBC shock."""
    
    # Set the frequency range for business cycles (6-32 quarters)
    low_freq = 1 / 32
    high_freq = 1 / 6
    
    # Step 2: Compute the spectral density matrix
    def compute_spectral_density(var_results, freq):
        """Compute spectral density matrix for the VAR model at a given frequency."""
        omega = 2 * np.pi * freq
        I = np.eye(var_results.neqs)
        A = np.sum(
            [var_results.coefs[i] * np.exp(-1j * omega * (i + 1)) for i in range(var_results.k_ar)],
            axis=0
        )
        return np.linalg.inv(I - A) @ var_results.sigma_u @ np.linalg.inv(I - A).conj().T
    
    # Step 3: Integrate variance contributions
    freqs = np.linspace(low_freq, high_freq, 500)
    spectral_density_sum = np.zeros((data.shape[1], data.shape[1]), dtype=complex)
    
    for freq in freqs:
        spectral_density_sum += compute_spectral_density(var_results, freq)
    
    # Normalize by the frequency band width
    spectral_density_sum /= len(freqs)
    
    # Step 4: Convert complex matrix to real before SVD
    spectral_density_real = np.real(spectral_density_sum)
    U, S, Vt = svd(spectral_density_real)
    
    # The first singular vector corresponds to the main business cycle shock
    mbc_shock = U[:, 0]
    
    # Step 5: Compute the contribution of the MBC shock to the variance
    mbc_contributions = np.real(np.dot(mbc_shock, spectral_density_sum @ mbc_shock))
    
    return mbc_shock, mbc_contributions

def constuctMBC_bandpass(residuals):
        
    from sklearn.decomposition import PCA
    from scipy.signal import butter, filtfilt
    
    # Step 2: Apply a band-pass filter to isolate business cycle frequencies
    def bandpass_filter(data, lowcut, highcut, fs, order=2):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data, axis=0)
    
    # Filter residuals (e.g., unemployment residuals)
    filtered_residuals = bandpass_filter(residuals, lowcut=1/32, highcut=1/6, fs=1)
    
    # Step 3: Use PCA to identify the dominant shock driving filtered unemployment
    pca = PCA(n_components=1)
    mbc_shock = pca.fit_transform(filtered_residuals)
    
    # Normalize the MBC shock
    mbc_shock = (mbc_shock - np.mean(mbc_shock)) / np.std(mbc_shock)
    
    return mbc_shock

def constructMBC_spectral(residuals):
    from scipy.signal import periodogram

    
    # Perform spectral analysis to isolate cyclical fluctuations
    # Compute the power spectral density of unemployment residuals
    freqs, power = periodogram(residuals['unemployment_rate'], scaling='density')
    
    # Focus on the business cycle frequency range (e.g., 6-32 quarters ~ 0.2-0.5 Hz)
    business_cycle_mask = (freqs >= 0.2) & (freqs <= 0.5)
    cyclical_power = np.sum(power[business_cycle_mask])
    
    # Identify the MBC shock as the component maximizing unemployment fluctuations
    mbc_shock = residuals @ np.linalg.pinv(residuals.T @ residuals)[:, 0]  # First principal component of residuals
    
    # Normalize the MBC shock
    mbc_shock = (mbc_shock - np.mean(mbc_shock)) / np.std(mbc_shock)
    
    return mbc_shock


