import matplotlib.pyplot as plt
import numpy as np


def plot_irfs(
    irfs: np.ndarray, impulse_index: int, response_index: int, figsize=(8, 5)
):
    """
    Plots impulse response functions for specified impulse and response indices.

    Parameters:
        irfs (numpy.ndarray): Array of orthogonalized IRFs with shape
                              (forecast_steps, num_variables, num_variables).
        impulse_index (int): Index of the variable applying the impulse.
        response_index (int): Index of the variable responding to the impulse.
        signif (numpy.ndarray, optional): Significance bands with shape
                                          (forecast_steps, num_variables, num_variables, 2)
                                          representing lower and upper bounds for responses.
        figsize (tuple, optional): Size of the figure (width, height). Defaults to (8, 5).
    """
    forecast_steps = irfs.shape[0]
    impulse_response = irfs[:, impulse_index, response_index]

    # Plot the impulse response
    plt.figure(figsize=figsize)
    plt.plot(
        range(forecast_steps),
        impulse_response,
        label=f"Impulse: {impulse_index}, Response: {response_index}",
        color="blue",
    )

    # Plot enhancements
    plt.axhline(
        0, color="black", linestyle="--", linewidth=0.8
    )  # Add a horizontal line at y=0
    plt.title(
        f"Impulse Response: Variable {response_index} to Impulse in Variable {impulse_index}"
    )
    plt.xlabel("Forecast Steps")
    plt.ylabel("Response")
    plt.legend()
    plt.grid(True)
    plt.show()
