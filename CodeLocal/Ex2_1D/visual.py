import torch
import matplotlib.pyplot as plt
import numpy as np


def plot_data(score1_np, score2_np, stateF_np):

    # Use a mask to split by class
    mask_pass = stateF_np.flatten() == 1
    mask_fail = stateF_np.flatten() == 0

    plt.scatter(
        score1_np[mask_pass],
        score2_np[mask_pass],
        s=8,
        color="blue",
        label="Victory (1)",
    )

    plt.scatter(
        score1_np[mask_fail],
        score2_np[mask_fail],
        s=8,
        color="red",
        label="Defeat (0)",
    )

    plt.xlabel("Score 1")
    plt.ylabel("Score 2")
    plt.title("Challenge Scores vs Outcome")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        "C:\\Documents\\600_Testing_Programing\\MLStudy\\CodeLocal\\Ex2_1D\\Plots\\DataScatter.png"
    )
    plt.close()


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

    # Plot
    plt.figure(figsize=(8, 5))
    plt.scatter(
        X_np,
        y_np,
        s=8,
        color="blue",
        label="Data",
        alpha=0.6,
    )
    plt.plot(
        x_vals,
        y_pred,
        color="red",
        linewidth=2,
        label="Sigmoid",
    )
    plt.axvline(
        boundary, color="green", linestyle="--", label=f"Boundary: x = {boundary:.2f}"
    )
    plt.xlabel("Normalized Average Score")
    plt.ylabel("Predicted Probability")
    plt.title("Logistic Classifier Decision Boundary")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        "C:\\Documents\\600_Testing_Programing\\MLStudy\\CodeLocal\\Ex2_1D\\Plots\\DecisionBoundary.png"
    )
    plt.close()
