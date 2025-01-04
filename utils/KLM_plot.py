import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2


class KLMConfidenceSet:
    def __init__(self, model, specification):
        """
        Initialize the KLMConfidenceSet object.

        Parameters:
        - model: Fitted model object (e.g., VAR, LP, etc.).
        - specification: String defining the specification of the model.
        """
        self.model = model
        self.specification = specification
        self.coef = self.model.params  # Extract coefficients [α_hat, β_hat]
        self.cov_matrix = self.model.cov_params()  # Covariance matrix


        self.gamma_f_range = np.linspace(-0.6, 0.2, 500)
        self.lambda_range = np.linspace(-0.6, 0.2, 500)
        self.gamma_f_grid, self.lambda_grid = np.meshgrid(
            self.gamma_f_range, self.lambda_range
        )
        self.critical_values = {
            "68%": chi2.ppf(0.68, df=2),
            "90%": chi2.ppf(0.90, df=2),
            "95%": chi2.ppf(0.95, df=2),
        }

    def klm_statistic(self, estimates):
        """
        Compute the KLM statistic for a given set of parameter estimates.

        Parameters:
        - estimates: Estimated coefficients [α_hat, β_hat].

        Returns:
        - KLM statistic value.
        """
        diff = estimates - self.coef
        return 0.5 * diff.T @ np.linalg.inv(self.cov_matrix) @ diff
    
    def compute_res()
        
    def compute_klm_grid(
        self,
    ):
        """
        Compute KLM statistics over a parameter grid.

        Returns:
        - klm_grid: Grid of KLM statistics for each combination of γ_f and λ.
        """
        klm_grid = np.zeros_like(self.gamma_f_grid)
        for i in range(self.gamma_f_grid.shape[0]):
            for j in range(self.gamma_f_grid.shape[1]):
                simulated_params = np.array(
                    [self.gamma_f_grid[i, j], self.lambda_grid[i, j]]
                )
                klm_grid[i, j] = self.klm_statistic(simulated_params)
        return klm_grid  # instead compute what depends on the parameter in KLM

    def plot_confidence_regions(self):
        """
        Plot the confidence regions for the KLM statistic.
        """
        klm_grid = self.compute_klm_grid()
        plt.figure(figsize=(8, 6))
        for level, crit_val in self.critical_values.items():
            plt.contour(
                self.gamma_f_grid,
                self.lambda_grid,
                klm_grid,
                levels=[crit_val],
                linewidths=1.5,
                label=f"{level} CI",
            )

        # Add point estimates
        plt.scatter(
            [self.coef[0]], [self.coef[1]], color="black", label="Point Estimate"
        )

        plt.xlabel(r"$\gamma_f$")
        plt.ylabel(r"$\lambda$")
        plt.title("Confidence Sets Based on KLM Statistic")
        plt.legend()
        plt.grid()
        plt.show()
