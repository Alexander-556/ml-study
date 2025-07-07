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
    plt.savefig("C:\\Documents\\600_Testing_Programing\\MLStudy\\CodeLocal\\Ex2\\Plots\\DataScatter.png")
    plt.close()

# def plot_decision_boundary(model, score1_np, score2_np):