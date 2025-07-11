# model.py
import torch
from typing import Optional


class Logistic2D:
    def __init__(self, learning_rate=0.01, epochs=10000):
        self.lr = learning_rate
        self.epochs = epochs
        self.w: Optional[torch.Tensor] = None  # Shape: (2, 1)
        self.b: Optional[torch.Tensor] = None  # Shape: (1,)
        self.loss_history = []

    # Function: sigmoid()
    # Define sigmoid function. This is the function that we're aiming to optimize
    def sigmoid(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        return 1 / (1 + torch.exp(-z))

    # Function: binary_cross_entropy()
    # Define the BCE Loss function used in our model. This measures our performance
    def binary_cross_entropy(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        eps = 1e-8  # Prevent log(0)
        return -torch.mean(
            y * torch.log(y_hat + eps) + (1 - y) * torch.log(1 - y_hat + eps)
        )

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        #Set random starting values
        self.w = torch.randn(2, 1, requires_grad=True)  # (2, 1)
        self.b = torch.randn(1, requires_grad=True)  # scalar

        #Begin training!
        for epoch in range(self.epochs):
            #Calculate our function, and our loss
            z = X @ self.w + self.b  # (n, 1)
            y_hat = self.sigmoid(z)  # (n, 1)
            loss = self.binary_cross_entropy(y_hat, y)
            
            #Calculate the gradient using the loss
            loss.backward()

            #Alter our values according to the loss
            with torch.no_grad():
                self.w -= self.lr * self.w.grad  # type: ignore
                self.b -= self.lr * self.b.grad  # type: ignore

                #Reset our gradients - otherwise torch appends them
                self.w.grad.zero_()  # type: ignore
                self.b.grad.zero_()  # type: ignore

            #Store our loss value. We can plot this later
            self.loss_history.append(loss.item())

            #Progress report for every hundred epochs
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

    #Gives us the probability of success for a point based on our current model
    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(X @ self.w + self.b)  # type: ignore

    #Makes a binary prediciton for success for a point based on our current model
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        proba = self.predict_proba(X)
        return (proba >= 0.5).float()

    #Gives us our coefficients if we want to report them
    def coefficients(self) -> tuple[float, float, float]:
        return self.w[0].item(), self.w[1].item(), self.b.item()  # type: ignore
