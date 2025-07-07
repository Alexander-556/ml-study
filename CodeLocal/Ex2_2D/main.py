# main.py
# The entry point of the Exercise 2.

"""Import Libraries and Custom Files"""

# Necessary Libraries
import numpy as np
import torch

# Custom Classes and Functions
from visual import plot_data
from model import LogisticGD


def main():
    # * Step 1: Load and prepare data
    # Load data file using absolute path
    # ! Replace the following path with your own
    data = np.loadtxt(
        "C:\\Documents\\600_Testing_Programing\\MLStudy\\Docs\\ex2data1.txt",
        delimiter=",",
    )

    # Store data as 1D numpy lists
    score1_np = data[:, 0].reshape(-1, 1)
    score2_np = data[:, 1].reshape(-1, 1)
    stateF_np = data[:, 2].reshape(-1, 1)

    # Convert from numpy list to torch tensors
    score1 = torch.from_numpy(score1_np).float()
    score2 = torch.from_numpy(score2_np).float()
    stateF = torch.from_numpy(stateF_np).float()

    # Shape: (n, 2)
    X = torch.cat([score1, score2], dim=1)  
    # Shape: (n, 1)
    y = stateF

    X_mean = X.mean(dim=0)
    X_std = X.std(dim=0)
    X = (X - X_mean) / X_std

    # * Step 2: Initialize and train model
    model = LogisticGD(learning_rate=0.1, epochs=1000)
    model.fit(X, y)

    # * Step 3: Output final parameters
    w1, w2, b = model.coefficients()
    print(f"Final Model: y = sigmoid({w1:.4f} * x1 + {w2:.4f} * x2 + {b:.4f})")

    # * Step 4: Visualization
    plot_data(score1_np, score2_np, stateF_np)


if __name__ == "__main__":
    main()
