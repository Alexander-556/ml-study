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
    plt.savefig("C:\\Documents\\600_Testing_Programing\\MLStudy\\CodeLocal\\Ex1_GD\\Plots\\FittedModel.png")
    plt.close()

def plot_loss(loss_history):
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, color='blue', linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Training Loss over Epochs")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("C:\\Documents\\600_Testing_Programing\\MLStudy\\CodeLocal\\Ex1_GD\\Plots\\LossHist.png")
    plt.close()

def plot_grad(grad_norm_history):
    plt.figure(figsize=(8, 5))
    plt.plot(grad_norm_history, color='blue', linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Gradient Norm")
    plt.title("Gradient Norm over Epochs")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("C:\\Documents\\600_Testing_Programing\\MLStudy\\CodeLocal\\Ex1_GD\\Plots\\GradNorm.png")
    plt.close()