"""CCGL's two-layer BiGRU encoder, projection head and predictor."""

from torch import cat, nn


def initialize(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.GRU):
        for name, parameter in module.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(parameter)
            elif "weight_hh" in name:
                nn.init.orthogonal_(parameter)
            else:
                nn.init.zeros_(parameter)


def projection(width, depth):
    return nn.Sequential(*[
        layer for _ in range(depth)
        for layer in (nn.Linear(width, width), nn.ReLU())
    ])


class Encoder(nn.Module):
    def __init__(self, embedding_dim=64, hidden_dim=128):
        super().__init__()
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=2,
                          batch_first=True, bidirectional=True)

    def forward(self, x):
        # Inputs have the same fixed 100-step zero padding as the reference model.
        _, hidden = self.gru(x)
        return cat((hidden[-2], hidden[-1]), dim=-1)


class ContrastiveModel(nn.Module):
    def __init__(self, embedding_dim=64, hidden_dim=128):
        super().__init__()
        self.encoder = Encoder(embedding_dim, hidden_dim)
        self.projection = projection(2 * hidden_dim, 4)
        self.apply(initialize)

    def forward(self, x):
        return self.projection(self.encoder(x))


class Predictor(nn.Module):
    def __init__(self, depth, embedding_dim=64, hidden_dim=128):
        super().__init__()
        self.encoder = Encoder(embedding_dim, hidden_dim)
        self.projection = projection(2 * hidden_dim, depth)
        self.regressor = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1), nn.Softplus(),
        )
        self.apply(initialize)

    def load_pretrained(self, pretrained):
        self.encoder.load_state_dict(pretrained.encoder.state_dict())
        self.projection.load_state_dict(
            pretrained.projection[:len(self.projection)].state_dict())

    def forward(self, x):
        return self.regressor(self.projection(self.encoder(x))).squeeze(-1)
