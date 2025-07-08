# model.py
import torch
from typing import Optional


class Logistic2D:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.w: Optional[torch.Tensor] = None  # Shape (2,)
        self.b: Optional[torch.Tensor] = None  # Scalar
        self.loss_history = []

    # Helper: _validate_gradients()
    # Validates the parameters.
    def _validate_gradients(self):
        if self.w is None or self.b is None:
            raise ValueError("Weights not initialized. Did you forget to call fit()?")

        if self.w.grad is None or self.b.grad is None:
            raise RuntimeError(
                "Gradients are not available. Did you run loss.backward()?"
            )

    def sigmoid(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:

        return 1 / (1 + torch.exp(-z))

    def binary_cross_entropy(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:

        eps = 1e-8  # avoid log(0)
        return -torch.mean(
            y * torch.log(y_hat + eps) + (1 - y) * torch.log(1 - y_hat + eps)
        )

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        n_samples, n_features = X.shape

        self.w = torch.zeros(n_features, requires_grad=True)
        self.b = torch.tensor(0.0, requires_grad=True)

        for epoch in range(self.epochs):
            z = X @ self.w + self.b
            y_hat = self.sigmoid(z)
            loss = self.binary_cross_entropy(y_hat, y)

            loss.backward()

            # Validate the gradients
            self._validate_gradients()

            with torch.no_grad():
                self.w -= self.lr * self.w.grad  # type: ignore
                self.b -= self.lr * self.b.grad  # type: ignore

                self.w.grad.zero_()  # type: ignore
                self.b.grad.zero_()  # type: ignore

            self.loss_history.append(loss.item())

            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.sigmoid(X @ self.w + self.b)  # type: ignore

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            proba = self.predict_proba(X)
            return (proba >= 0.5).float()

    def coefficients(self) -> tuple[float, float, float]:
        return self.w[0].item(), self.w[1].item(), self.b.item()  # type: ignore
