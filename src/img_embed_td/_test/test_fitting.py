import torch
import torch.nn as nn

from img_embed_td.fitting import training_loop


class MockTripletDataset:
    """Mock dataset for testing that returns fixed triplets."""

    def __init__(self, num_samples: int = 10, input_dim: int = 8):
        self.num_samples = num_samples
        self.input_dim = input_dim

        # Generate fixed random data for reproducibility
        torch.manual_seed(42)
        self.data = [
            (
                torch.randn(input_dim),  # anchor
                torch.randn(input_dim),  # positive
                torch.randn(input_dim),  # negative
            )
            for _ in range(num_samples)
        ]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]


class SimpleEmbeddingModel(nn.Module):
    """Simple linear model for testing."""

    def __init__(self, input_dim: int = 8, embedding_dim: int = 4):
        super().__init__()
        self.fc = nn.Linear(input_dim, embedding_dim)

    def forward(self, x):
        return self.fc(x)


def simple_triplet_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """Simple triplet loss for testing."""
    pos_dist = torch.sum((anchor - positive).square())
    neg_dist = torch.sum((anchor - negative).square())
    loss = torch.relu(pos_dist - neg_dist + margin)
    return loss


def test_training_loop():
    """Test that training_loop runs successfully and updates model parameters."""
    # Setup
    input_dim = 8
    embedding_dim = 4
    num_samples = 10
    epochs = 2

    # Create mock dataset and model
    dataset = MockTripletDataset(num_samples=num_samples, input_dim=input_dim)
    model = SimpleEmbeddingModel(input_dim=input_dim, embedding_dim=embedding_dim)

    # Store initial parameters to verify they change
    initial_params = [p.clone() for p in model.parameters()]

    # Create optimizer and run training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Run training loop
    training_loop(
        dataset=dataset,
        model=model,
        epochs=epochs,
        loss_func=simple_triplet_loss,
        opt=optimizer,
    )

    # Verify model parameters have changed (indicating training occurred)
    params_changed = False
    for initial, current in zip(initial_params, model.parameters()):
        if not torch.allclose(initial, current):
            params_changed = True
            break

    assert params_changed, "Model parameters should change after training"
    print(" test_training_loop passed: model parameters updated during training")


if __name__ == "__main__":
    test_training_loop()
