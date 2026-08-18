from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np


class BaseBenchmarkModel:
    """Minimal train/predict interface shared by every benchmark model."""

    def __init__(self, config: dict) -> None:
        self.config = config

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> "BaseBenchmarkModel":
        raise NotImplementedError

    def predict(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def count_params(self) -> int:
        return 0

    def save(self, path: Path) -> None:
        with path.open("wb") as f:
            pickle.dump(self, f)


class TrendPriorW1Model(BaseBenchmarkModel):
    """Predict the next point as the last observed target value."""

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> "TrendPriorW1Model":
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        horizon = int(self.config["prediction_horizon"])
        last = x[:, -1, 0]
        return np.repeat(last.reshape(-1, 1), horizon, axis=1)


class TrendPriorW2Model(BaseBenchmarkModel):
    """Extrapolate a per-window least-squares trend into the next point."""

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> "TrendPriorW2Model":
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        horizon = int(self.config["prediction_horizon"])
        steps = np.arange(x.shape[1], dtype=float)
        future = np.arange(x.shape[1], x.shape[1] + horizon, dtype=float)
        preds = []
        for row in x[:, :, 0]:
            slope, intercept = np.polyfit(steps, row, 1)
            preds.append(intercept + slope * future)
        return np.asarray(preds, dtype=float)


class ELMModel(BaseBenchmarkModel):
    """Extreme Learning Machine with fixed random hidden layer and least-squares head."""

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> "ELMModel":
        rng = np.random.default_rng(int(self.config.get("seed", 42)))
        hidden = int(self.config.get("hidden_size", 32))
        flat_x = x_train.reshape(len(x_train), -1)
        self.x_mean = flat_x.mean(axis=0, keepdims=True)
        self.x_std = np.where(flat_x.std(axis=0, keepdims=True) < 1e-8, 1.0, flat_x.std(axis=0, keepdims=True))
        z = (flat_x - self.x_mean) / self.x_std
        self.win = rng.normal(scale=1.0, size=(z.shape[1], hidden))
        self.bias = rng.normal(scale=0.1, size=(1, hidden))
        hidden_x = self._activation(z @ self.win + self.bias)
        self.wout = np.linalg.pinv(hidden_x) @ y_train
        return self

    def _activation(self, values: np.ndarray) -> np.ndarray:
        activation = self.config.get("activation", "relu")
        if activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-values))
        return np.maximum(values, 0.0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        flat_x = x.reshape(len(x), -1)
        z = (flat_x - self.x_mean) / self.x_std
        return self._activation(z @ self.win + self.bias) @ self.wout

    def count_params(self) -> int:
        return int(np.prod(self.win.shape) + np.prod(self.bias.shape) + np.prod(self.wout.shape))


class TorchSequenceModel(BaseBenchmarkModel):
    """Optional PyTorch backend for neural sequence models."""

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.model = None
        self.x_scaler_mean = 0.0
        self.x_scaler_std = 1.0
        self.y_scaler_mean = 0.0
        self.y_scaler_std = 1.0

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> "TorchSequenceModel":
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for this model backend") from exc

        seed = int(self.config.get("seed", 42))
        torch.manual_seed(seed)
        np.random.seed(seed)
        x_train = np.asarray(x_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32)
        self.x_scaler_mean = float(np.mean(x_train))
        self.x_scaler_std = float(np.std(x_train) or 1.0)
        self.y_scaler_mean = float(np.mean(y_train))
        self.y_scaler_std = float(np.std(y_train) or 1.0)
        x_train = (x_train - self.x_scaler_mean) / self.x_scaler_std
        y_train = (y_train - self.y_scaler_mean) / self.y_scaler_std

        self.model = _TorchNet(self.config)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.config.get("learning_rate", 1e-3)),
            weight_decay=float(self.config.get("weight_decay", 0.0)),
        )
        criterion = nn.MSELoss()
        batch_size = int(self.config.get("batch_size", 32))
        epochs = int(self.config.get("epochs", 20))
        x_tensor = torch.from_numpy(x_train)
        y_tensor = torch.from_numpy(y_train)
        for _ in range(epochs):
            order = torch.randperm(len(x_tensor))
            for start in range(0, len(order), batch_size):
                idx = order[start : start + batch_size]
                pred = self.model(x_tensor[idx])
                loss = criterion(pred, y_tensor[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fitted")
        import torch

        x = np.asarray(x, dtype=np.float32)
        x = (x - self.x_scaler_mean) / self.x_scaler_std
        with torch.no_grad():
            pred = self.model(torch.from_numpy(x)).numpy()
        return pred * self.y_scaler_std + self.y_scaler_mean

    def count_params(self) -> int:
        if self.model is None:
            return 0
        return int(sum(p.numel() for p in self.model.parameters()))


def _torch_modules() -> Any:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    return torch, nn, F


class _TorchNet:
    """Factory wrapper so torch is imported only when requested."""

    def __new__(cls, config: dict):
        torch, nn, F = _torch_modules()
        architecture = config["architecture"].lower()
        horizon = int(config["prediction_horizon"])
        hidden = int(config.get("hidden_size", 64))
        layers = int(config.get("num_layers", 1))
        dropout = float(config.get("dropout", 0.0))
        input_size = int(config.get("input_size", 1))
        nhead = max(1, int(config.get("nhead", 4)))
        num_experts = max(1, int(config.get("num_experts", 4)))

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.architecture = architecture
                bidirectional = architecture in {"bilstm", "bigru"}
                if architecture in {"lstm", "bilstm"}:
                    self.sequence = nn.LSTM(input_size, hidden, layers, batch_first=True, bidirectional=bidirectional)
                    out_dim = hidden * (2 if bidirectional else 1)
                    self.head = nn.Linear(out_dim, horizon)
                elif architecture in {"gru", "bigru"}:
                    self.sequence = nn.GRU(input_size, hidden, layers, batch_first=True, bidirectional=bidirectional)
                    out_dim = hidden * (2 if bidirectional else 1)
                    self.head = nn.Linear(out_dim, horizon)
                elif architecture == "rnn":
                    self.sequence = nn.RNN(input_size, hidden, layers, batch_first=True)
                    self.head = nn.Linear(hidden, horizon)
                elif architecture in {"mlp", "dnn"}:
                    width = int(config.get("dense_width", hidden))
                    self.sequence = nn.Sequential(
                        nn.Flatten(),
                        nn.LazyLinear(width),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(width, width),
                        nn.ReLU(),
                        nn.Linear(width, horizon),
                    )
                    self.head = None
                elif architecture in {"cnn_lstm", "cnn_gru"}:
                    self.conv = nn.Conv1d(input_size, hidden, kernel_size=3, padding=1)
                    recurrent = nn.LSTM if architecture == "cnn_lstm" else nn.GRU
                    self.sequence = recurrent(hidden, hidden, layers, batch_first=True)
                    self.head = nn.Linear(hidden, horizon)
                elif architecture == "cnn1d":
                    self.sequence = nn.Sequential(
                        nn.Conv1d(input_size, 16, kernel_size=3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool1d(2),
                        nn.Conv1d(16, 32, kernel_size=3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool1d(2),
                        nn.Conv1d(32, 64, kernel_size=3, padding=1),
                        nn.ReLU(),
                        nn.AdaptiveAvgPool1d(1),
                        nn.Flatten(),
                        nn.Linear(64, hidden),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden, horizon),
                    )
                    self.head = None
                elif architecture == "tcn":
                    self.sequence = nn.Sequential(
                        nn.Conv1d(input_size, hidden, kernel_size=3, padding=2, dilation=2),
                        nn.ReLU(),
                        nn.Conv1d(hidden, hidden, kernel_size=3, padding=4, dilation=4),
                        nn.ReLU(),
                    )
                    self.head = nn.Linear(hidden, horizon)
                elif architecture in {"transformer", "informer"}:
                    self.proj = nn.Linear(input_size, hidden)
                    heads = max(1, min(nhead, hidden))
                    while hidden % heads != 0 and heads > 1:
                        heads -= 1
                    layer = nn.TransformerEncoderLayer(
                        d_model=hidden,
                        nhead=heads,
                        dim_feedforward=hidden * 4,
                        dropout=dropout,
                        batch_first=True,
                    )
                    self.sequence = nn.TransformerEncoder(layer, num_layers=layers)
                    self.head = nn.Linear(hidden, horizon)
                elif architecture == "attmoe":
                    heads = max(1, min(nhead, hidden))
                    while hidden % heads != 0 and heads > 1:
                        heads -= 1
                    self.proj = nn.Linear(input_size, hidden)
                    self.attn = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
                    self.experts = nn.ModuleList([nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU()) for _ in range(num_experts)])
                    self.gate = nn.Linear(hidden, num_experts)
                    self.dropout = nn.Dropout(dropout)
                    self.head = nn.Linear(hidden, horizon)
                elif architecture in {"slstm", "mlstm", "xlstm"}:
                    recurrent = nn.LSTM if architecture != "mlstm" else nn.GRU
                    self.sequence = recurrent(input_size, hidden, layers, batch_first=True)
                    self.norm = nn.LayerNorm(hidden)
                    self.gate = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())
                    self.head = nn.Linear(hidden, horizon)
                else:
                    raise ValueError(f"Unsupported torch architecture: {architecture}")

            def forward(self, x):
                if self.architecture in {"mlp", "dnn"}:
                    return self.sequence(x)
                if self.architecture == "cnn1d":
                    return self.sequence(x.transpose(1, 2))
                if self.architecture in {"cnn_lstm", "cnn_gru"}:
                    z = self.conv(x.transpose(1, 2)).transpose(1, 2)
                    z, _ = self.sequence(z)
                    return self.head(z[:, -1, :])
                if self.architecture == "tcn":
                    z = self.sequence(x.transpose(1, 2))[:, :, -1]
                    return self.head(z)
                if self.architecture in {"transformer", "informer"}:
                    z = self.sequence(self.proj(x))
                    return self.head(z[:, -1, :])
                if self.architecture == "attmoe":
                    z = self.dropout(self.proj(x))
                    z, _ = self.attn(z, z, z)
                    pooled = z[:, -1, :]
                    weights = torch.softmax(self.gate(pooled), dim=-1)
                    expert_stack = torch.stack([expert(pooled) for expert in self.experts], dim=1)
                    mixed = torch.sum(weights.unsqueeze(-1) * expert_stack, dim=1)
                    return self.head(mixed)
                if self.architecture in {"slstm", "mlstm", "xlstm"}:
                    z, _ = self.sequence(x)
                    z = self.norm(z[:, -1, :])
                    z = z * self.gate(z)
                    return self.head(z)
                z, _ = self.sequence(x)
                return self.head(z[:, -1, :])

        return Net()


class ExternalGradientBoostingModel(BaseBenchmarkModel):
    """Optional XGBoost/LightGBM backend fitted one regressor per horizon."""

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> "ExternalGradientBoostingModel":
        backend = self.config["backend"]
        if backend == "xgboost":
            try:
                from xgboost import XGBRegressor
            except ImportError as exc:
                raise RuntimeError("xgboost is required for this model backend") from exc
            factory = lambda: XGBRegressor(
                n_estimators=int(self.config.get("n_estimators", 100)),
                max_depth=int(self.config.get("max_depth", 3)),
                learning_rate=float(self.config.get("learning_rate", 0.05)),
                subsample=float(self.config.get("subsample", 1.0)),
                colsample_bytree=float(self.config.get("colsample_bytree", 1.0)),
                random_state=int(self.config.get("seed", 42)),
                n_jobs=1,
                verbosity=0,
            )
        elif backend == "lightgbm":
            try:
                from lightgbm import LGBMRegressor
            except ImportError as exc:
                raise RuntimeError("lightgbm is required for this model backend") from exc
            factory = lambda: LGBMRegressor(
                n_estimators=int(self.config.get("n_estimators", 100)),
                learning_rate=float(self.config.get("learning_rate", 0.05)),
                random_state=int(self.config.get("seed", 42)),
                n_jobs=1,
                verbosity=-1,
                force_col_wise=True,
            )
        else:
            raise ValueError(f"Unsupported external backend: {backend}")
        flat_x = x_train.reshape(len(x_train), -1)
        self.models = []
        for step in range(y_train.shape[1]):
            model = factory()
            model.fit(flat_x, y_train[:, step])
            self.models.append(model)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        flat_x = x.reshape(len(x), -1)
        return np.column_stack([model.predict(flat_x) for model in self.models])


def build_model(config: dict) -> BaseBenchmarkModel:
    backend = config["backend"]
    if backend in {"trend_prior_l1", "persistence"}:
        return TrendPriorW1Model(config)
    if backend in {"trend_prior_l2", "linear_trend"}:
        return TrendPriorW2Model(config)
    if backend == "elm":
        return ELMModel(config)
    if backend == "torch":
        return TorchSequenceModel(config)
    if backend in {"xgboost", "lightgbm"}:
        return ExternalGradientBoostingModel(config)
    raise ValueError(f"Unknown model backend: {backend}")
