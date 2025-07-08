# main.py
# The entry point of the Exercise 2.

"""Import Libraries and Custom Files"""

# Necessary Libraries
import numpy as np
import torch

# Custom Classes and Functions
from visual import plot_data, plot_sigmoid_boundary
from model import Logistic1D


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

    # Calculate average from the two scores
    # and store avg into a 1D numpy list
    avg_score_np = (score1_np + score2_np) / 2

    # Convert from numpy list to torch tensors
    avg_score = torch.from_numpy(avg_score_np).float()
    stateF = torch.from_numpy(stateF_np).float()

    # Normalize Inputs using Z score
    avg_mean = avg_score.mean()
    avg_std = avg_score.std()
    avg_score = (avg_score - avg_mean) / avg_std

    # * Step 2: Initialize and train model
    model = Logistic1D(learning_rate=0.01, epochs=5000)
    model.fit(avg_score, stateF)

    # * Step 3: Output final parameters
    # Grab coefficients and piece up the model
    w, b = model.coefficients()
    print(f"Final Model: y = sigmoid({w:.4f} * x + {b:.4f})")

    # Calculate the accuracy values
    preds = model.predict(avg_score)
    acc = (preds == stateF).float().mean().item()
    print(f"Accuracy: {acc:.4f}")

    # * Step 4: Visualization
    plot_data(score1_np, score2_np, stateF_np)
    plot_sigmoid_boundary(model, avg_score, stateF)


if __name__ == "__main__":
    main()
