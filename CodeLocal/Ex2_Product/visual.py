# visual.py
# Plot the given data and the predicted values.

"""Import Libraries and Custom Files"""
# Necessary Libraries
import torch
import matplotlib.pyplot as plt
import numpy as np


def plot_data(score1_np, score2_np, stateF_np):

    # Use a mask to split by class
    mask_pass = stateF_np.flatten() == 1
    mask_fail = stateF_np.flatten() == 0

    #Plot the victory points in blue
    plt.scatter(
        score1_np[mask_pass],
        score2_np[mask_pass],
        s=8,
        color="blue",
        label="Victory (1)",
    )

    #Plot the defeat points in red
    plt.scatter(
        score1_np[mask_fail],
        score2_np[mask_fail],
        s=8,
        color="red",
        label="Defeat (0)",
    )

    #Some label for the plot
    plt.xlabel("Score 1")
    plt.ylabel("Score 2")
    plt.title("Challenge Scores vs Outcome")
    plt.legend()
    plt.grid(True)
    plt.savefig("./CodeLocal/Ex2_Product/Plots/DataScatter.png")
    plt.close()

#This plots our solution curve
def plot_sigmoid_boundary(model, X_tensor, y_tensor):
    X_np = X_tensor.numpy().flatten()
    y_np = y_tensor.numpy().flatten()

    # Sort X for smoother curve
    x_vals = np.linspace(X_np.min() - 1, X_np.max() + 1, 200)
    x_tensor = torch.tensor(x_vals).unsqueeze(1).float()

    with torch.no_grad():
        y_pred = model.predict_proba(x_tensor).numpy()

    # Decision boundary
    w, b = model.w.item(), model.b.item()
    boundary = -b / w

    #Plot the data along success or failure
    plt.figure(figsize=(8, 5))
    plt.scatter(
        X_np,
        y_np,
        s=8,
        color="blue",
        label="Data",
        alpha=0.6,
    )

    #Plot our sigmoid on top of the data
    plt.plot(
        x_vals,
        y_pred,
        color="red",
        linewidth=2,
        label="Sigmoid",
    )
    #This plots the boundary - the point at which we switch from predicting 
    #success to failure - over the sigmoid and the data
    plt.axvline(
        boundary, 
        color="green", 
        linestyle="--", 
        label=f"Boundary: x = {boundary:.2f}"
    )

    #Some labels for the graph
    plt.xlabel("Normalized Average Score")
    plt.ylabel("Predicted Probability")
    plt.title("Logistic Classifier Decision Boundary")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./CodeLocal/Ex2_Product/Plots/DecisionBoundary.png")
    plt.close()
