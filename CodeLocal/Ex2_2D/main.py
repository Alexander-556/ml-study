# main.py
# The entry point of the Exercise 2, 2D approach.

"""Import Libraries and Custom Files"""

# Necessary Libraries
import numpy as np
import torch

# Custom Classes and Functions
from visual import plot_data
from model import Logistic2D
from utils import normalize


def main():
    # * Step 1: Load and prepare data
    # ! Replace the following path with your own
    data = np.loadtxt(
        "./Docs/ex2data1.txt",
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

    # Normalize Inputs using Z score
    norm_score1 = normalize(score1)
    norm_score2 = normalize(score2)

    norm_X = torch.cat([norm_score1, norm_score2], dim=1)

    # * Step 2: Initialize and train model
    model = Logistic2D(learning_rate=0.01, epochs=5000)
    model.fit(norm_X, stateF)

    # * Step 3: Output final parameters
    # Grab coefficients and piece up the model
    w1, w2, b = model.coefficients()
    print(f"Final Model: y = sigmoid({w1:.4f} * x1 + {w2:.4f} * x2 + {b:.4f})")

    # Calculate the accuracy values
    preds = model.predict(norm_X)
    acc = (preds == stateF).float().mean().item()
    print(f"Accuracy: {acc:.4f}")

    # * Step 4: Visualization
    # plot_data(score1_np, score2_np, stateF_np)
    # plot_sigmoid_boundary(model, norm_X, stateF)


if __name__ == "__main__":
    main()
