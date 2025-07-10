# main.py
# The entry point of the Exercise 2. A more effective 1D approach

"""Import Libraries and Custom Files"""

# Necessary Libraries
import numpy as np
import torch

# Custom Classes and Functions
# from visual import plot_data
from model import Logistic1DProd
from utils import normalize
from visual import plot_data, plot_sigmoid_boundary


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
    score_prod_np = score1_np * score2_np
    #Reviewing the given data, taking the product of the scores represents the trend well

    # Convert from numpy list to torch tensors
    score_prod = torch.from_numpy(score_prod_np).float()
    stateF = torch.from_numpy(stateF_np).float()

    # Normalize Inputs using Z score
    norm_score_prod = normalize(score_prod)

    # * Step 2: Initialize and train model
    #Epochs are the number of times we run the fit, while learning rate 
    #scales how much we change our weights and biases each epoch. Tweaking
    #these values can allow us to adjust the model and improve performance
    model = Logistic1DProd(learning_rate=0.01, epochs=5000)
    model.fit(norm_score_prod, stateF)

    # * Step 3: Output final parameters
    # Grab coefficients and piece up the model
    w, b = model.coefficients()
    print(f"Final Model: y = sigmoid({w:.4f} * x + {b:.4f})") #Our final equation

    # Calculate the accuracy values
    preds = model.predict(norm_score_prod)
    acc = (preds == stateF).float().mean().item()
    print(f"Accuracy: {acc:.4f}")

    # * Step 4: Visualization
    plot_data(score1_np, score2_np, stateF_np)
    plot_sigmoid_boundary(model, norm_score_prod, stateF)


if __name__ == "__main__":
    main()