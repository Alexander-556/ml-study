import torch
from typing import Optional

class Logistic1D:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.w: Optional[torch.Tensor] = None  # Scalar weight
        self.b: Optional[torch.Tensor] = None  # Scalar bias
        self.loss_history = []
    
    # Helper: _validate_gradients()
    # Validates the parameters.
    def _validate_gradients(self):
        if self.w is None or self.b is None:
            raise ValueError(
                "Weights not initialized. Did you forget to call fit()?"
                )

        if self.w.grad is None or self.b.grad is None:
            raise ValueError(
                "Gradients are not available. Did you run loss.backward()?"
            )

    def sigmoid(self, z: torch.Tensor) -> torch.Tensor:
        return 1 / (1 + torch.exp(-z))

    def binary_cross_entropy(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        eps = 1e-8  # Prevent log(0)
        return -torch.mean(
            y * torch.log(y_hat + eps) + (1 - y) * torch.log(1 - y_hat + eps)
        )

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        # Initialize weights
        self.w = torch.randn(1, requires_grad=True)
        self.b = torch.randn(1, requires_grad=True)

        for epoch in range(self.epochs):
            z = X * self.w + self.b
            y_hat = self.sigmoid(z)
            loss = self.binary_cross_entropy(y_hat, y)

            loss.backward()

            self._validate_gradients()

            with torch.no_grad():
                self.w -= self.lr * self.w.grad # type: ignore
                self.b -= self.lr * self.b.grad # type: ignore
                self.w.grad.zero_() # type: ignore
                self.b.grad.zero_() # type: ignore

            self.loss_history.append(loss.item())
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(X * self.w + self.b) # type: ignore

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return (self.predict_proba(X) >= 0.5).float()

    def coefficients(self) -> tuple[float, float]:
        return self.w.item(), self.b.item() # type: ignore
