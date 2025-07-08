# model.py
# The class definition of LinearNormal Class, Gradient Descent approach.

"""Import Libraries and Custom Files"""

# Necessary Libraries
import torch
from typing import Optional

class LinearGD:
    # Class Constructor
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.w: Optional[torch.Tensor] = None
        self.b: Optional[torch.Tensor] = None
        self.loss_history = []
        self.grad_norm_history = []

    # Helper: _validate_gradients()
    # Validates the parameters.
    def _validate_gradients(self):
        if self.w is None or self.b is None:
            raise ValueError("Weights not initialized. Did you forget to call fit()?")

        if self.w.grad is None or self.b.grad is None:
            raise RuntimeError("Gradients are not available. Did you run loss.backward()?")

    # Function: fit()
    # Fit model using gradient descent.
    def fit(self, x: torch.Tensor, y: torch.Tensor):

        # Grab the length of the 1D Array
        m = x.shape[0]

        # Initialize coefficient to start at random
        self.w = torch.randn(1, requires_grad=True)
        self.b = torch.randn(1, requires_grad=True)

        # Start Epoch training
        for epoch in range(self.epochs):

            # Calculated predicted y for performance eval
            y_pred = x * self.w + self.b
            # Define loss function
            loss = (1 / m) * torch.sum((y_pred - y) ** 2)

            # Start backwards pass
            # pytorch calculates the gradient values
            loss.backward()

            # Validate the gradients
            self._validate_gradients()

            grad_norm = torch.sqrt(
                self.w.grad.pow(2) + self.b.grad.pow(2) # type: ignore
                )
            
            self.grad_norm_history.append(grad_norm.item())

            # * Temporarily disable gradient tracking when 
            # * updating the params and resetting the gradient
            with torch.no_grad():
                # Update params and reset gradients for each
                self.w -= self.lr * self.w.grad # type: ignore
                self.b -= self.lr * self.b.grad # type: ignore
                self.w.grad.zero_() # type: ignore 
                self.b.grad.zero_() # type: ignore

            # Logs the loss history
            self.loss_history.append(loss.item())

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}, w: {self.w.item():.4f}, b: {self.b.item():.4f}")

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        if self.w is None or self.b is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        return x * self.w + self.b

    def coefficients(self) -> tuple[float, float]:
        if self.w is None or self.b is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        return self.w.item(), self.b.item()
    
