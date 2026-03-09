"""Neural model definitions used by DMN sequence strategies."""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMPositionNet(nn.Module):
    """Single-layer LSTM head producing a position in [-1, 1]."""

    def __init__(
        self,
        n_features: int,
        hidden: int = 32,
        dropout: float = 0.1,
        use_ticker_embedding: bool = True,
        n_tickers: int | None = None,
        ticker_emb_dim: int = 8,
    ):
        super().__init__()
        if use_ticker_embedding and (n_tickers is None or n_tickers <= 0):
            raise ValueError("n_tickers must be set when use_ticker_embedding=True.")

        self.use_ticker_embedding = use_ticker_embedding
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        if self.use_ticker_embedding:
            self.ticker_emb = nn.Embedding(n_tickers, ticker_emb_dim)
            self.fc = nn.Linear(hidden + ticker_emb_dim, 1)
        else:
            self.fc = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, ticker_id: torch.Tensor | None = None) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = self.drop(out[:, -1, :])
        if self.use_ticker_embedding:
            if ticker_id is None:
                raise ValueError("ticker_id is required when use_ticker_embedding=True.")
            out = torch.cat([out, self.ticker_emb(ticker_id)], dim=-1)
        pos = torch.tanh(self.fc(out))
        return pos.squeeze(-1)


class VLSTMPositionNet(nn.Module):
    """VLSTM approximation: Variable Selection Network followed by LSTM."""

    def __init__(
        self,
        n_features: int,
        hidden: int = 32,
        dropout: float = 0.1,
        use_ticker_embedding: bool = True,
        n_tickers: int | None = None,
        ticker_emb_dim: int = 8,
    ):
        super().__init__()
        if use_ticker_embedding and (n_tickers is None or n_tickers <= 0):
            raise ValueError("n_tickers must be set when use_ticker_embedding=True.")

        self.use_ticker_embedding = use_ticker_embedding
        self.vsn = nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.ReLU(),
            nn.Linear(n_features, n_features),
        )
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        if self.use_ticker_embedding:
            self.ticker_emb = nn.Embedding(n_tickers, ticker_emb_dim)
            self.fc = nn.Linear(hidden + ticker_emb_dim, 1)
        else:
            self.fc = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, ticker_id: torch.Tensor | None = None) -> torch.Tensor:
        gate_scores = self.vsn(x)
        gate_weights = torch.softmax(gate_scores, dim=-1)
        x_sel = gate_weights * x

        out, _ = self.lstm(x_sel)
        out = self.drop(out[:, -1, :])
        if self.use_ticker_embedding:
            if ticker_id is None:
                raise ValueError("ticker_id is required when use_ticker_embedding=True.")
            out = torch.cat([out, self.ticker_emb(ticker_id)], dim=-1)
        pos = torch.tanh(self.fc(out))
        return pos.squeeze(-1)


class xLSTMPositionNet(nn.Module):
    """Lightweight sLSTM-style xLSTM block with exponential gates."""

    def __init__(
        self,
        n_features: int,
        hidden: int = 32,
        dropout: float = 0.1,
        eps: float = 1e-8,
        use_ticker_embedding: bool = True,
        n_tickers: int | None = None,
        ticker_emb_dim: int = 8,
    ):
        super().__init__()
        if use_ticker_embedding and (n_tickers is None or n_tickers <= 0):
            raise ValueError("n_tickers must be set when use_ticker_embedding=True.")

        self.hidden = hidden
        self.eps = eps
        self.use_ticker_embedding = use_ticker_embedding

        self.wf = nn.Linear(n_features, hidden)
        self.wi = nn.Linear(n_features, hidden)
        self.wo = nn.Linear(n_features, hidden)
        self.wz = nn.Linear(n_features, hidden)

        self.rf = nn.Linear(hidden, hidden, bias=False)
        self.ri = nn.Linear(hidden, hidden, bias=False)
        self.ro = nn.Linear(hidden, hidden, bias=False)
        self.rz = nn.Linear(hidden, hidden, bias=False)

        self.drop = nn.Dropout(dropout)
        if self.use_ticker_embedding:
            self.ticker_emb = nn.Embedding(n_tickers, ticker_emb_dim)
            self.fc = nn.Linear(hidden + ticker_emb_dim, 1)
        else:
            self.fc = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, ticker_id: torch.Tensor | None = None) -> torch.Tensor:
        b, t, _ = x.shape
        h = torch.zeros((b, self.hidden), device=x.device)
        c = torch.zeros((b, self.hidden), device=x.device)
        n = torch.zeros((b, self.hidden), device=x.device)

        for k in range(t):
            xt = x[:, k, :]

            f_tilde = self.wf(xt) + self.rf(h)
            i_tilde = self.wi(xt) + self.ri(h)
            o = torch.sigmoid(self.wo(xt) + self.ro(h))
            z = torch.tanh(self.wz(xt) + self.rz(h))

            f_hat = torch.exp(torch.clamp(f_tilde, max=10.0))
            i_hat = torch.exp(torch.clamp(i_tilde, max=10.0))
            denom = f_hat + i_hat + self.eps
            f = f_hat / denom
            i = i_hat / denom

            c = f * c + i * z
            n = f * n + i
            h = o * (c / (n + self.eps))

        out = self.drop(h)
        if self.use_ticker_embedding:
            if ticker_id is None:
                raise ValueError("ticker_id is required when use_ticker_embedding=True.")
            out = torch.cat([out, self.ticker_emb(ticker_id)], dim=-1)
        pos = torch.tanh(self.fc(out))
        return pos.squeeze(-1)
