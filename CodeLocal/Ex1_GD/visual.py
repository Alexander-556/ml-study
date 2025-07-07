import matplotlib.pyplot as plt
import numpy as np


def plot_regression(x_np, y_np, y_pred_np):
    plt.scatter(
        x_np,
        y_np,
        s=8,
        label="Data",
    )

    plt.plot(
        x_np,
        y_pred_np,
        color="red",
        label="Fitted Line",
    )
    
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Gradient Descent Fit")
    plt.legend()
    plt.grid(True)
    plt.show()
