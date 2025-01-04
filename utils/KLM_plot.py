import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import chi2
from matplotlib.patches import Ellipse


# Extract coefficients and covariance matrix
coef = model.params  # [α_hat, β_hat]
cov_matrix = model.cov_params()  # Covariance matrix

# Step 3: Simulate Sampling Distribution of Coefficients
n_simulations = 1000
simulated_coefs = np.random.multivariate_normal(coef, cov_matrix, size=n_simulations)

# Step 4: Compute KLM Statistic for Each Simulation
def klm_statistic(estimates, mean, cov):
    diff = estimates - mean
    return 0.5 * diff.T @ np.linalg.inv(cov) @ diff

klm_values = [klm_statistic(sim, coef, cov_matrix) for sim in simulated_coefs]

# Step 5: Compute KLM Percentiles
klm_threshold_68 = chi2.ppf(0.68, df=2)
klm_threshold_90 = chi2.ppf(0.90, df=2)
klm_threshold_95 = chi2.ppf(0.95, df=2)

# Step 6: Plot β vs α with KLM Percentiles
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(simulated_coefs[:, 1], simulated_coefs[:, 0],
            color='skyblue', alpha=0.5, label='Simulated Coefficients')
ax.scatter(coef[1], coef[0], color='black', marker='x', s=100, label='Estimated Coefficients')

# Add Confidence Ellipses for 68%, 90%, 95%
def draw_confidence_ellipse(mean, cov, ax, percentile, color, label):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    width, height = 2 * np.sqrt(vals * chi2.ppf(percentile, df=2))
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    ellipse = Ellipse(xy=mean[::-1], width=width, height=height, angle=angle,
                      edgecolor=color, facecolor='none', lw=2, label=label)
    ax.add_patch(ellipse)

draw_confidence_ellipse(coef, cov_matrix, ax, 0.68, 'green', '68% Confidence')
draw_confidence_ellipse(coef, cov_matrix, ax, 0.90, 'orange', '90% Confidence')
draw_confidence_ellipse(coef, cov_matrix, ax, 0.95, 'red', '95% Confidence')

plt.title('KLM-Based Confidence Region for OLS Coefficients ($\lambda$ and $\gamma_f$)')
plt.xlabel('$\lambda$ Coefficient')
plt.ylabel('$\gamma_f$ Coefficient')
plt.legend()
plt.show()
