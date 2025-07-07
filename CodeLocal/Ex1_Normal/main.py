# main.py
# The entry point of the Exercise 1 Normal Equation Approach.

"""Import Libraries and Custom Files"""

# Necessary Libraries
import numpy as np
import torch

# Custom Classes and Functions
from model import LinearNormal
from visual import plot_regression


def main():
    # * Step 1: Load and prepare data
    # Load data file using absolute path
    # ! Replace the following path with your own
    data = np.loadtxt("C:\\Documents\\600_Testing_Programing\\MLStudy\\Docs\\ex1data1.txt", delimiter=",")
    # Store data as 1D numpy lists
    x_np = data[:, 0].reshape(-1, 1)
    y_np = data[:, 1].reshape(-1, 1)
    # Convert from numpy list to torch tensors
    x = torch.from_numpy(x_np).float()
    y = torch.from_numpy(y_np).float()

    # * Step 2: Perform Model Fit
    # Call fit class
    model = LinearNormal()
    # Call fit function
    model.fit(x, y)

    # * Step 3: Plot Results
    # Predict
    y_pred = model.predict(x)
    # Grab coefficients
    w, b = model.coefficients()
    # Print results
    print(f"Analytical Solution: y = {w:.4f} * x + {b:.4f}")
    # Plot
    plot_regression(x_np, y_np, y_pred.detach().numpy())

if __name__ == "__main__":
    main()
