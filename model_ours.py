from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np


class TrendPriorPool:
    """Trend Prior Pool: an extensible registry of zero-parameter interpretable
    trend predictors that independently extrapolate H-step forecasts from a SOH window.

    The pool encapsulates physically meaningful base predictors (Trend_Prior_l1,
    Trend_Prior_l2,
    and potential future additions like exponential decay or physics-informed priors)
    as callable functions. The Stage-aware Selector then dynamically blends these priors
    via degradation-stage-conditioned weights.

    Design rationale (aligned with main.tex §3.2.2):
      - **Zero parameters**: each prior is a closed-form extrapolation rule
      - **Interpretability**: each prior has a clear physical meaning
      - **Extensibility**: new priors can be registered via `register()`
      - **Independence**: each prior operates on the window independently

    Usage::
        pool = TrendPriorPool()
        pool.register("Trend_Prior_l1", lambda values, horizon: ...)
        pool.register("Trend_Prior_l2", lambda values, horizon: ...)
        predictions = pool.predict_all(values, horizon)
        # -> {"Trend_Prior_l1": array, "Trend_Prior_l2": array}
    """

    def __init__(self) -> None:
        self._priors: dict[str, callable] = {}
        self._prior_order: list[str] = []  # Maintain insertion order for selector compatibility

    def register(self, name: str, prior_fn: callable) -> None:
        """Register a trend prior predictor.

        Args:
            name: unique identifier for this prior (e.g. "Trend_Prior_l1")
            prior_fn: callable with signature (values: ndarray[N, T], horizon: int) -> ndarray[N, horizon]
                     where values[:, t] are the SOH observations in the window
        """
        if name in self._priors:
            raise ValueError(f"Trend prior '{name}' is already registered")
        self._priors[name] = prior_fn
        self._prior_order.append(name)

    def predict_all(self, values: np.ndarray, horizon: int) -> dict[str, np.ndarray]:
        """Generate predictions from all registered priors.

        Args:
            values: shape (N, window_size), SOH observations
            horizon: number of steps to extrapolate

        Returns:
            dict mapping prior names to predictions, shape (N, horizon) each
        """
        return {name: fn(values, horizon) for name, fn in self._priors.items()}

    def predict_stacked(self, values: np.ndarray, horizon: int) -> np.ndarray:
        """Generate predictions from all priors, stacked for selector blending.

        Args:
            values: shape (N, window_size)
            horizon: forecast horizon

        Returns:
            shape (N, n_priors, horizon), where axis=1 follows self._prior_order
        """
        preds = [self._priors[name](values, horizon) for name in self._prior_order]
        return np.stack(preds, axis=1)  # (N, n_priors, horizon)

    def prior_names(self) -> list[str]:
        """Return the ordered list of registered prior names."""
        return list(self._prior_order)

    def count_priors(self) -> int:
        """Return the number of registered priors."""
        return len(self._priors)

    @staticmethod
    def create_default_pool() -> "TrendPriorPool":
        """Factory: create a pool with the default ARCF priors.

        These two priors cover the observed battery degradation regimes:
          - **Trend_Prior_l1** (T_p = x_T): optimal for near-stationary SOH plateaus
          - **Trend_Prior_l2** (T_l = a·t + b): optimal for steady linear decay

        Future extensions (noted in main.tex): exponential decay, double-exponential,
        physics-informed capacity fade models.
        """
        pool = TrendPriorPool()

        def trend_prior_l1(values: np.ndarray, horizon: int) -> np.ndarray:
            """Hold the last observed value flat over the horizon."""
            last = values[:, -1]  # (N,)
            return np.repeat(last[:, None], horizon, axis=1)  # (N, horizon)

        def trend_prior_l2(values: np.ndarray, horizon: int) -> np.ndarray:
            """OLS linear fit on the window, extrapolated `horizon` steps ahead."""
            window = values.shape[1]
            steps = np.arange(window, dtype=float)
            centered = steps - steps.mean()
            denom = float(np.sum(centered**2)) or 1.0
            mean = values.mean(axis=1)
            slope = ((values - mean[:, None]) * centered[None, :]).sum(axis=1) / denom
            intercept = mean  # value at the window center (t = mean(steps))
            # Future step offsets relative to the window center
            future = (window - 1 - steps.mean()) + np.arange(1, horizon + 1, dtype=float)
            return intercept[:, None] + slope[:, None] * future[None, :]  # (N, horizon)

        pool.register("Trend_Prior_l1", trend_prior_l1)
        pool.register("Trend_Prior_l2", trend_prior_l2)
        return pool


class StageAwareTrendSelector:
    """Stage-aware trend selector: degradation-stage-conditioned blending of
    trend priors from a Trend Prior Pool.

    Architecture (aligned with main.tex §3.2):
      1. **Trend Prior Pool**: generates candidate trend predictions (Trend_Prior_l1,
         Trend_Prior_l2, etc.)
      2. **Stage-aware Selector**: learns degradation-stage-dependent weights via softmax gate

        L_t = Σ_i w_i(z_t) * T_i,   where Σ w_i = 1

    The selector weights ``w = softmax(W z_t + b)`` are produced from aging features
    ``z_t`` derived from the SOH window (level, slope, volatility, cumulative drop, mean level).
    These window-derived proxies stand in for the degradation state since the literal
    ``[cycle, SOH, capacity, resistance]`` state is not available in the input tensor.

    The gate is trained (Adam, full-batch NumPy) to minimize MSE of the blended trend ``L_t``
    against the target ``y``. The residual TCN downstream then learns whatever nonlinear
    structure ``L_t`` leaves behind.

    The pool-based architecture supports extensibility: future priors (exponential decay,
    physics-informed models) can be added without changing the selector logic.
    """

    N_FEATURES = 6  # z_t dimensionality (5 aging features + bias handled separately)

    def __init__(
        self,
        trend_pool: TrendPriorPool | None = None,
        hidden: int = 0,
        epochs: int = 300,
        learning_rate: float = 0.05,
        l2: float = 1e-4,
        seed: int = 42,
    ) -> None:
        self.trend_pool = trend_pool if trend_pool is not None else TrendPriorPool.create_default_pool()
        self.n_priors = self.trend_pool.count_priors()
        if self.n_priors == 0:
            raise ValueError("TrendPriorPool must have at least one registered prior")

        self.hidden = int(hidden)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.l2 = float(l2)
        self.seed = int(seed)

        # Gate parameters. When hidden == 0: z_t -> n_priors logits (linear layer)
        # When hidden > 0: z_t -> hidden (tanh) -> n_priors logits (1-hidden-layer MLP)
        self.w1: np.ndarray | None = None  # (n_feat, hidden) or (n_feat, n_priors)
        self.b1: np.ndarray | None = None
        self.w2: np.ndarray | None = None  # (hidden, n_priors) when hidden > 0
        self.b2: np.ndarray | None = None

        # Standardisation stats for the aging features.
        self._feat_mean: np.ndarray | None = None
        self._feat_std: np.ndarray | None = None
        self.horizon = 1

    # ---- public API (drop-in for the former linear branch) -----------------
    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> "StageAwareTrendSelector":
        values = np.asarray(x_train, dtype=float)[:, :, 0]
        target = np.asarray(y_train, dtype=float)
        if target.ndim == 1:
            target = target.reshape(-1, 1)
        self.horizon = target.shape[1]

        z = self._aging_features(values)
        self._feat_mean = z.mean(axis=0)
        self._feat_std = z.std(axis=0)
        self._feat_std[self._feat_std == 0.0] = 1.0
        z_std = (z - self._feat_mean) / self._feat_std

        base = self.trend_pool.predict_stacked(values, self.horizon)  # (N, n_priors, horizon)
        self._init_gate(z_std.shape[1])
        self._train_gate(z_std, base, target)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.w1 is None:
            raise RuntimeError("Stage-aware trend selector has not been fitted")
        values = np.asarray(x, dtype=float)[:, :, 0]
        weights = self._gate_weights(values)  # (N, n_priors)
        base = self.trend_pool.predict_stacked(values, self.horizon)  # (N, n_priors, horizon)
        return np.einsum("nc,nch->nh", weights, base)

    def weights(self, x: np.ndarray) -> np.ndarray:
        """Per-sample selector weights for each prior in the pool, Σ=1.

        Returns:
            shape (N, n_priors), where columns correspond to self.trend_pool.prior_names()
        """
        values = np.asarray(x, dtype=float)[:, :, 0]
        return self._gate_weights(values)

    def count_params(self) -> int:
        params = [p for p in (self.w1, self.b1, self.w2, self.b2) if p is not None]
        return int(sum(int(np.prod(p.shape)) for p in params))

    # ---- aging features (z_t) ---------------------------------------------
    @staticmethod
    def _aging_features(values: np.ndarray) -> np.ndarray:
        """Aging state proxies derived from the SOH window.

        Columns: [last SOH level, within-window slope, volatility (diff std),
        cumulative drop (last - first), mean level]. A degrading cell shows a
        low level, negative slope and larger cumulative drop; an early/plateau
        cell shows a high level and near-zero slope.
        """
        window = values.shape[1]
        steps = np.arange(window, dtype=float)
        centered = steps - steps.mean()
        denom = float(np.sum(centered**2)) or 1.0
        mean = values.mean(axis=1)
        first = values[:, 0]
        last = values[:, -1]
        slope = ((values - mean[:, None]) * centered[None, :]).sum(axis=1) / denom
        diff = np.diff(values, axis=1)
        if diff.shape[1] == 0:
            diff = np.zeros((len(values), 1), dtype=float)
        volatility = diff.std(axis=1)
        drop = last - first
        return np.column_stack([last, slope, volatility, drop, mean])

    # ---- softmax gate ------------------------------------------------------
    def _init_gate(self, n_feat: int) -> None:
        rng = np.random.default_rng(self.seed)
        scale = 0.1
        if self.hidden > 0:
            self.w1 = rng.normal(0.0, scale, size=(n_feat, self.hidden))
            self.b1 = np.zeros(self.hidden, dtype=float)
            self.w2 = rng.normal(0.0, scale, size=(self.hidden, self.n_priors))
            self.b2 = np.zeros(self.n_priors, dtype=float)
        else:
            self.w1 = rng.normal(0.0, scale, size=(n_feat, self.n_priors))
            self.b1 = np.zeros(self.n_priors, dtype=float)
            self.w2 = None
            self.b2 = None

    def _logits(self, z_std: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        if self.hidden > 0:
            hidden = np.tanh(z_std @ self.w1 + self.b1)
            logits = hidden @ self.w2 + self.b2
            return logits, hidden
        return z_std @ self.w1 + self.b1, None

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    def _gate_weights(self, values: np.ndarray) -> np.ndarray:
        z = self._aging_features(values)
        z_std = (z - self._feat_mean) / self._feat_std
        logits, _ = self._logits(z_std)
        return self._softmax(logits)

    def _train_gate(self, z_std: np.ndarray, base: np.ndarray, target: np.ndarray) -> None:
        """Full-batch Adam on MSE of the blended trend vs. target.

        Gradients flow only through the softmax gate; the base predictions are
        fixed. dL/dlogits is derived analytically via the softmax Jacobian.
        """
        n = len(z_std)
        lr = self.learning_rate
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        params = {"w1": self.w1, "b1": self.b1}
        if self.hidden > 0:
            params["w2"], params["b2"] = self.w2, self.b2
        m = {k: np.zeros_like(v) for k, v in params.items()}
        v = {k: np.zeros_like(v) for k, v in params.items()}

        for step in range(1, self.epochs + 1):
            logits, hidden = self._logits(z_std)
            weights = self._softmax(logits)  # (N, n_priors)
            pred = np.einsum("nc,nch->nh", weights, base)  # (N, horizon)
            err = pred - target  # (N, horizon)

            # dL/dw_c = (2/(N*H)) * sum_h err_h * base[:, c, h]
            hcount = target.shape[1]
            dpred = (2.0 / (n * hcount)) * err  # (N, horizon)
            dweights = np.einsum("nh,nch->nc", dpred, base)  # (N, n_priors)
            # softmax Jacobian: dlogits = w * (dweights - sum_c(dweights*w))
            dot = np.sum(dweights * weights, axis=1, keepdims=True)
            dlogits = weights * (dweights - dot)  # (N, 2)

            grads: dict[str, np.ndarray] = {}
            if self.hidden > 0:
                grads["w2"] = hidden.T @ dlogits + self.l2 * self.w2
                grads["b2"] = dlogits.sum(axis=0)
                dhidden = (dlogits @ self.w2.T) * (1.0 - hidden**2)
                grads["w1"] = z_std.T @ dhidden + self.l2 * self.w1
                grads["b1"] = dhidden.sum(axis=0)
            else:
                grads["w1"] = z_std.T @ dlogits + self.l2 * self.w1
                grads["b1"] = dlogits.sum(axis=0)

            for k in params:
                g = grads[k]
                m[k] = beta1 * m[k] + (1 - beta1) * g
                v[k] = beta2 * v[k] + (1 - beta2) * (g * g)
                mhat = m[k] / (1 - beta1**step)
                vhat = v[k] / (1 - beta2**step)
                params[k] -= lr * mhat / (np.sqrt(vhat) + eps)

        self.w1, self.b1 = params["w1"], params["b1"]
        if self.hidden > 0:
            self.w2, self.b2 = params["w2"], params["b2"]

    def state(self) -> dict[str, Any]:
        """Serialisable gate parameters + feature stats."""
        def _tolist(a: np.ndarray | None):
            return None if a is None else a.tolist()

        return {
            "strategy": "stage_aware",
            "hidden": self.hidden,
            "w1": _tolist(self.w1),
            "b1": _tolist(self.b1),
            "w2": _tolist(self.w2),
            "b2": _tolist(self.b2),
            "feat_mean": _tolist(self._feat_mean),
            "feat_std": _tolist(self._feat_std),
            "horizon": self.horizon,
        }


class TrendPriorW1:
    """Parameter-free trend used by the RMSE-optimized one-step ARCF path.

    This is a lightweight wrapper that directly uses the Trend_Prior_l1 prior from the pool
    without the overhead of selector training. Used when trend_strategy="Trend_Prior_l1"
    or when auto mode selects Trend_Prior_l1 for horizon=1.
    """

    def __init__(self) -> None:
        self.horizon = 1
        self._pool = TrendPriorPool.create_default_pool()

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> "TrendPriorW1":
        target = np.asarray(y_train)
        self.horizon = target.shape[1] if target.ndim > 1 else 1
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=float)[:, :, 0]
        return self._pool._priors["Trend_Prior_l1"](values, self.horizon)

    def weights(self, x: np.ndarray) -> np.ndarray:
        """Compatibility view: all weight assigned to Trend_Prior_l1 (index 0)."""
        n_samples = len(x)
        # Return (N, 2) for backward compatibility with 2-prior assumption in plotting code
        return np.column_stack([np.ones(n_samples), np.zeros(n_samples)])

    def count_params(self) -> int:
        return 0

    def state(self) -> dict[str, Any]:
        return {"strategy": "Trend_Prior_l1", "horizon": self.horizon}


class OursModel:
    """Adaptive Residual Compensation Framework (ARCF): Trend Prior Pool with
    Stage-aware Selector + residual TCN + scalar residual-gain strategy.

    Architecture (aligned with main.tex):
      1. **Trend Prior Pool + Stage-aware Selector**: dynamically blend interpretable
         trend priors (Trend_Prior_l1, Trend_Prior_l2) based on degradation stage
      2. **Residual TCN**: learn nonlinear structure unexplained by the trend
      3. **Scalar alpha strategy**: calibrate residual gain on validation data

    Flow:
      x -> Trend Prior Pool -> {T_Trend_Prior_l1, T_Trend_Prior_l2, ...}
        -> Stage-aware Selector(z_t) -> L_t = Σ w_i * T_i
        -> residual_gt = y - L_t
        -> TCN learns residual from raw window x -> Residual
        -> alpha strategy (analytic regularized least-squares by default)
        -> SOH = L_t + alpha * Residual + validation_bias

    ``trend_strategy=auto`` uses parameter-free Trend_Prior_l1 for one-step
    prediction (lower RMSE in ablation), and stage-aware blend for longer horizons.
    ``stage_aware`` remains available explicitly for ablation and interpretability.

    **Alpha strategies** (scalar calibration on validation data):
      - ``analytic`` (ARCF default): ridge-regularized LS with prior centered at alpha=1
        Gives lowest cross-dataset RMSE (0.009578 vs fixed_one 0.009587)
      - ``fixed_one``: use complete learned residual (alpha=1)
      - ``grid``: validation-MSE grid search over [0, alpha_max]
      - ``fixed_half`` / ``fixed_zero``: ablation baselines

    **Note**: The learned Gate strategy has been removed from this implementation.
    Experimental results (notebook §5.7) showed Gate achieved RMSE=0.010780,
    significantly worse than scalar strategies (analytic=0.009578). The 737-parameter
    Gate added complexity without performance gain, so only scalar strategies are
    retained for production and ablation experiments.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.seed = int(config.get("seed", 42))
        self.horizon = int(config.get("prediction_horizon", 1))
        configured_trend = str(config.get("trend_strategy", "auto")).lower()
        if configured_trend in {"persistence", "trend_prior_l1"}:
            configured_trend = "trend_prior_l1"
        self.trend_strategy = (
            "trend_prior_l1" if configured_trend == "auto" and self.horizon == 1
            else "stage_aware" if configured_trend == "auto"
            else configured_trend
        )
        if self.trend_strategy == "trend_prior_l1":
            self.trend_selector = TrendPriorW1()
        elif self.trend_strategy == "stage_aware":
            self.trend_selector = StageAwareTrendSelector(
                trend_pool=TrendPriorPool.create_default_pool(),
                hidden=int(config.get("selector_hidden", 0)),
                epochs=int(config.get("selector_epochs", 300)),
                learning_rate=float(config.get("selector_lr", 0.05)),
                l2=float(config.get("selector_l2", 1e-4)),
                seed=self.seed,
            )
        else:
            raise ValueError(f"Unsupported trend_strategy: {configured_trend}")

        self.tcn_model: Any = None
        self.best_params: dict[str, Any] = {}

        # Residual gain alpha: final = L_t + alpha * TCN_residual
        # Scalar strategies only (Gate removed per §5.7 ablation results)
        self.alpha_strategy = str(config.get("alpha_strategy", "analytic")).lower()
        if self.alpha_strategy == "gate":
            raise ValueError(
                "Gate strategy has been removed. Experimental results showed "
                "Gate RMSE=0.010780 vs analytic=0.009578. Use 'analytic', 'fixed_one', "
                "'grid', 'fixed_half', or 'fixed_zero' instead."
            )
        self.best_weight = 1.0

        # Residual debiasing: additive offset chosen on validation
        self.residual_debias = bool(config.get("residual_debias", False))
        self.residual_bias = 0.0

        # Per-window input normalization for the TCN
        self.window_norm = bool(config.get("tcn_window_norm", False))

        self.search_history: list[dict[str, Any]] = []
        self.loss_history: list[dict[str, Any]] = []
        self.fusion_history: list[dict[str, Any]] = []
        self.alpha_search_history: list[dict[str, Any]] = []

        # Validation arrays cached at fit time
        self._val_y: np.ndarray | None = None
        self._val_arima: np.ndarray | None = None
        self._val_residual: np.ndarray | None = None

        self.x_mean = 0.0
        self.x_std = 1.0
        self.y_mean = 0.0
        self.y_std = 1.0
        self.device_name = "cpu"

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> "OursModel":
        x_train = np.asarray(x_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32)
        x_val = np.asarray(x_val, dtype=np.float32)
        y_val = np.asarray(y_val, dtype=np.float32)

        # 1) Trend selector: fit the stage-aware blended linear trend L_t.
        self.trend_selector.fit(x_train, y_train)

        # 2) Residuals = what the linear trend failed to explain. The TCN's target
        #    is the residual, computed from the raw window x on both splits.
        r_train = (y_train - self.trend_selector.predict(x_train)).astype(np.float32)
        r_val = (y_val - self.trend_selector.predict(x_val)).astype(np.float32)

        self.x_mean = float(np.mean(self._anchor(x_train)))
        self.x_std = float(np.std(self._anchor(x_train)) or 1.0)
        # Scale the residual target (near-zero mean, small scale).
        self.y_mean = float(np.mean(r_train))
        self.y_std = float(np.std(r_train) or 1.0)

        # 3) Hyperparameter search — candidates are judged by how well the TCN
        #    fits the validation residual.
        candidates = self._candidate_configs()
        search_train_x, search_train_r = self._subsample(x_train, r_train, int(self.config.get("max_search_train_samples", 512)), self.seed)
        search_val_x, search_val_r = self._subsample(x_val, r_val, int(self.config.get("max_search_val_samples", 256)), self.seed + 17)
        best_loss = math.inf
        best_cfg = candidates[0]
        for idx, cfg in enumerate(candidates):
            model, _ = self._fit_tcn(search_train_x, search_train_r, search_val_x, search_val_r, cfg, int(self.config.get("candidate_epochs", 2)), record_loss=False)
            pred = self._predict_tcn(model, search_val_x)
            val_loss = _mse(search_val_r, pred)
            row = {"phase": "candidate", "candidate_id": idx, "validation_mse": val_loss, **cfg}
            self.search_history.append(row)
            if val_loss < best_loss:
                best_loss = val_loss
                best_cfg = cfg
        self.best_params = dict(best_cfg)

        # 4) Final TCN trained on the full residual target (scalar-alpha path).
        self.tcn_model, final_losses = self._fit_tcn(x_train, r_train, x_val, r_val, best_cfg, int(self.config.get("final_epochs", 6)), record_loss=True)
        self.loss_history.extend(final_losses)

        # 5) Cache validation branch outputs, then calibrate the residual gain
        #    alpha (per strategy) plus an optional debias offset.
        self._val_y = y_val
        self._val_arima = self.trend_selector.predict(x_val)
        self._val_residual = self._predict_tcn(self.tcn_model, x_val)
        self.best_weight, self.residual_bias = self._calibrate_residual(self.alpha_strategy, record=True)

        # 6) Record the residual-correction effect on validation (for inspection):
        #    trend-only error vs. final SOH error.
        self._record_residual_effect(x_val, y_val, r_val)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """SOH prediction using the calibrated scalar alpha + debias offset."""
        return self.predict_with_alpha(x, self.best_weight, self.residual_bias)

    def predict_with_alpha(self, x: np.ndarray, alpha: float, bias: float = 0.0) -> np.ndarray:
        """Prediction at a given (scalar) residual gain: L_t(x) + alpha * TCN(x) + bias."""
        x = np.asarray(x, dtype=np.float32)
        trend_pred = self.trend_selector.predict(x)
        residual_pred = self._predict_tcn(self.tcn_model, x)
        return trend_pred + float(alpha) * residual_pred + float(bias)

    def components(self, x: np.ndarray) -> dict[str, np.ndarray]:
        """Decompose the prediction into Trend / Residual / alpha / SOH_pred.

        Returns per-sample arrays for interpretability (the four-curve decomposition,
        trend/residual analysis). alpha is the calibrated scalar broadcast to all samples.
        """
        x = np.asarray(x, dtype=np.float32)
        trend_pred = self.trend_selector.predict(x)
        residual_pred = self._predict_tcn(self.tcn_model, x)
        alpha = np.full((len(x), 1), float(self.best_weight), dtype=float)
        soh_pred = trend_pred + float(self.best_weight) * residual_pred + float(self.residual_bias)
        return {"trend": trend_pred, "residual": residual_pred, "alpha": alpha, "soh_pred": soh_pred}

    def evaluate_alpha_strategies(self, x: np.ndarray, y: np.ndarray) -> list[dict[str, Any]]:
        """Score every alpha strategy on an external split (e.g. test).

        Reuses the single trained TCN, so all strategies are compared fairly and
        cheaply. Each strategy is applied exactly as the deployed method would be
        (alpha gain + optional debias). Returns one row per strategy.
        """
        x = np.asarray(x, dtype=np.float32)
        trend_pred = self.trend_selector.predict(x)
        residual_pred = self._predict_tcn(self.tcn_model, x)
        rows: list[dict[str, Any]] = []
        for strat in self.available_strategies():
            alpha, bias = self._calibrate_residual(strat, record=False)
            pred = trend_pred + float(alpha) * residual_pred + float(bias)
            rows.append({"strategy": strat, "alpha": float(alpha), "bias": float(bias), "mse": _mse(y, pred)})
        return rows

    @staticmethod
    def available_strategies() -> list[str]:
        return ["fixed_zero", "fixed_half", "fixed_one", "analytic", "grid"]

    def _calibrate_residual(self, strategy: str, record: bool = True) -> tuple[float, float]:
        """Resolve the residual gain alpha and an optional debias offset on val."""
        alpha = self._select_alpha(strategy, record=record)
        bias = 0.0
        if self.residual_debias and self._val_y is not None:
            # offset that makes the hybrid prediction zero-mean-error on validation
            resid_after = np.asarray(self._val_y - (self._val_arima + alpha * self._val_residual), dtype=float)
            bias = float(np.mean(resid_after)) if resid_after.size else 0.0
        return float(alpha), bias

    def _select_alpha(self, strategy: str, record: bool = True) -> float:
        """Resolve the residual gain alpha for a given strategy on validation data.

        Strategies:
          fixed_zero  -> 0.0  (pure stage-selected trend; residual-branch ablation)
          fixed_half  -> 0.5  (half-strength residual correction)
          fixed_one   -> 1.0  (pure residual addition)
          analytic    -> ridge-regularized least-squares optimum, clipped to [0, alpha_max]
          grid        -> validation-MSE grid search over [0, alpha_max]

        analytic is shrunk toward ``alpha_prior`` with strength ``alpha_l2`` to
        avoid overfitting alpha to a single validation cell.
        """
        y = self._val_y
        arima = self._val_arima
        residual = self._val_residual
        strategy = str(strategy).lower()
        alpha_max = float(self.config.get("alpha_max", 1.5))

        if strategy in ("fixed_zero", "zero"):
            return 0.0
        if strategy in ("fixed_half", "half"):
            return 0.5
        if strategy in ("fixed_one", "one", "fixed"):
            return 1.0

        if y is None or arima is None or residual is None:
            return 1.0

        if strategy == "analytic":
            # Ridge-regularized: minimize ||r - a*res||^2 + lambda*(a - prior)^2
            #   -> a = (<r,res> + lambda*prior) / (<res,res> + lambda)
            r = np.asarray(y - arima, dtype=float).ravel()
            res = np.asarray(residual, dtype=float).ravel()
            lam = float(self.config.get("alpha_l2", 0.0))
            prior = float(self.config.get("alpha_prior", 1.0))
            denom = float(res @ res) + lam
            alpha = (float(r @ res) + lam * prior) / denom if denom > 0 else 1.0
            return float(np.clip(alpha, 0.0, alpha_max))

        # default: grid search
        grid = np.linspace(0.0, alpha_max, int(self.config.get("alpha_grid_size", 31)))
        best_alpha, best_loss = 1.0, math.inf
        for a in grid:
            loss = _mse(y, arima + float(a) * residual)
            if record:
                self.alpha_search_history.append({"phase": "alpha_grid", "alpha": float(a), "validation_mse": loss})
            if loss < best_loss:
                best_loss, best_alpha = loss, float(a)
        return best_alpha

    def count_params(self) -> int:
        tcn_params = 0
        if self.tcn_model is not None:
            tcn_params = int(sum(p.numel() for p in self.tcn_model.parameters()))
        return tcn_params + self.trend_selector.count_params()

    def save(self, path: Path) -> None:
        import torch

        payload = {
            "model_family": "ARCF-TCN",
            "best_params": self.best_params,
            "best_weight": self.best_weight,
            "residual_bias": self.residual_bias,
            "alpha_strategy": self.alpha_strategy,
            "trend_strategy": self.trend_strategy,
            "window_norm": self.window_norm,
            "trend_selector": self.trend_selector.state(),
            "x_mean": self.x_mean,
            "x_std": self.x_std,
            "y_mean": self.y_mean,
            "y_std": self.y_std,
            "state_dict": None if self.tcn_model is None else self.tcn_model.state_dict(),
        }
        torch.save(payload, path)

    def _candidate_configs(self) -> list[dict[str, Any]]:
        rng = np.random.default_rng(self.seed)
        channels = _parse_int_choices(self.config.get("channel_choices", "16,24,32,48"))
        kernels = _parse_int_choices(self.config.get("kernel_choices", "2,3,5"))
        batch_sizes = _parse_int_choices(self.config.get("batch_choices", "32,64"))
        count = int(self.config.get("bo_initial_points", 3)) + int(self.config.get("bo_iterations", 2))
        candidates: list[dict[str, Any]] = []
        for i in range(max(1, count)):
            candidates.append(
                {
                    "channels": int(channels[i % len(channels)] if i < len(channels) else rng.choice(channels)),
                    "kernel_size": int(kernels[(i + 1) % len(kernels)] if i < len(kernels) else rng.choice(kernels)),
                    "batch_size": int(batch_sizes[i % len(batch_sizes)]),
                    "learning_rate": float(np.exp(rng.uniform(np.log(float(self.config.get("learning_rate_min", 5e-4))), np.log(float(self.config.get("learning_rate_max", 4e-3)))))),
                    "dropout": float(rng.uniform(float(self.config.get("dropout_min", 0.0)), float(self.config.get("dropout_max", 0.25)))),
                    "weight_decay": float(rng.uniform(0.0, float(self.config.get("weight_decay_max", 5e-4)))),
                }
            )
        return candidates

    def _fit_tcn(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        cfg: dict[str, Any],
        epochs: int,
        record_loss: bool,
    ) -> tuple[Any, list[dict[str, Any]]]:
        import torch
        import torch.nn as nn

        torch.set_num_threads(int(self.config.get("torch_num_threads", 1)))
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        device = torch.device("cuda" if torch.cuda.is_available() and bool(self.config.get("use_gpu", True)) else "cpu")
        self.device_name = str(device)
        model = _TCNForecastNet(
            input_size=x_train.shape[2],
            horizon=self.horizon,
            channels=int(cfg["channels"]),
            kernel_size=int(cfg["kernel_size"]),
            dropout=float(cfg["dropout"]),
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
        criterion = nn.MSELoss()
        x_tensor = torch.from_numpy(self._scale_x(x_train)).float()
        y_tensor = torch.from_numpy(self._scale_y(y_train)).float()
        val_x_tensor = torch.from_numpy(self._scale_x(x_val)).float().to(device)
        val_y_tensor = torch.from_numpy(self._scale_y(y_val)).float().to(device)
        batch_size = int(cfg["batch_size"])
        losses: list[dict[str, Any]] = []
        for epoch in range(1, epochs + 1):
            model.train()
            order = torch.randperm(len(x_tensor))
            epoch_losses: list[float] = []
            for start in range(0, len(order), batch_size):
                idx = order[start : start + batch_size]
                bx = x_tensor[idx].to(device)
                by = y_tensor[idx].to(device)
                pred = model(bx)
                loss = criterion(pred, by)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(self.config.get("grad_clip", 1.0)))
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))
            if record_loss:
                model.eval()
                with torch.no_grad():
                    val_loss = float(criterion(model(val_x_tensor), val_y_tensor).detach().cpu())
                losses.append({"epoch": epoch, "train_loss": float(np.mean(epoch_losses)), "validation_loss": val_loss, **cfg})
        return model, losses

    def _predict_tcn(self, model: Any, x: np.ndarray) -> np.ndarray:
        if model is None:
            raise RuntimeError("TCN branch has not been fitted")
        import torch

        device = next(model.parameters()).device
        model.eval()
        preds: list[np.ndarray] = []
        batch_size = int(self.config.get("predict_batch_size", 512))
        with torch.no_grad():
            for start in range(0, len(x), batch_size):
                bx = torch.from_numpy(self._scale_x(x[start : start + batch_size])).float().to(device)
                pred = model(bx).detach().cpu().numpy()
                preds.append(self._unscale_y(pred))
        return np.vstack(preds) if preds else np.empty((0, self.horizon), dtype=float)

    def _anchor(self, x: np.ndarray) -> np.ndarray:
        """Optionally subtract each window's last value (shape, not level)."""
        x = np.asarray(x, dtype=np.float32)
        if not self.window_norm or x.size == 0:
            return x
        return x - x[:, -1:, :]

    def _scale_x(self, x: np.ndarray) -> np.ndarray:
        return (self._anchor(x) - self.x_mean) / self.x_std

    def _scale_y(self, y: np.ndarray) -> np.ndarray:
        return (np.asarray(y, dtype=np.float32) - self.y_mean) / self.y_std

    def _unscale_y(self, y: np.ndarray) -> np.ndarray:
        return np.asarray(y, dtype=float) * self.y_std + self.y_mean

    def _record_residual_effect(self, x_val: np.ndarray, y_val: np.ndarray, r_val: np.ndarray) -> None:
        """Record how much the TCN residual branch improves over the trend alone.

        Populates ``fusion_history`` with the validation MSE of the trend-only
        prediction vs. the final trend+TCN prediction, so the notebook can show
        the effect of the residual-learning stage.
        """
        trend_pred = self.trend_selector.predict(x_val)
        residual_pred = self._predict_tcn(self.tcn_model, x_val)
        # Final prediction uses the deployed path (scalar alpha + debias),
        # so the reported effect matches predict().
        hybrid_pred = self.predict(x_val)
        trend_mse = _mse(y_val, trend_pred)
        hybrid_mse = _mse(y_val, hybrid_pred)
        # Also report how well the TCN fit the residual itself (target vs. pred).
        residual_fit_mse = _mse(r_val, residual_pred)
        self.fusion_history.append({"phase": "residual_effect", "stage": "阶段选择趋势", "stage_id": 0, "validation_mse": trend_mse})
        self.fusion_history.append({"phase": "residual_effect", "stage": "阶段趋势+TCN残差", "stage_id": 1, "validation_mse": hybrid_mse})
        self.residual_fit_mse = residual_fit_mse
        # trend_val_mse is the trend-only validation MSE. arima_val_mse kept as an
        # alias for any external reader still expecting the old attribute name.
        self.trend_val_mse = trend_mse
        self.arima_val_mse = trend_mse
        self.hybrid_val_mse = hybrid_mse

    @staticmethod
    def _subsample(x: np.ndarray, y: np.ndarray, limit: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        if len(x) <= limit:
            return x, y
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(x), size=limit, replace=False))
        return x[idx], y[idx]


class _CausalBlock:
    def __new__(cls, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
        import torch.nn as nn

        padding = (kernel_size - 1) * dilation

        class Block(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.padding = padding
                self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
                self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
                self.dropout = nn.Dropout(dropout)
                self.activation = nn.ReLU()
                self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

            def _trim(self, value):
                return value[:, :, : -self.padding] if self.padding > 0 else value

            def forward(self, x):
                out = self._trim(self.conv1(x))
                out = self.dropout(self.activation(out))
                out = self._trim(self.conv2(out))
                out = self.dropout(self.activation(out))
                return self.activation(out + self.downsample(x))

        return Block()


class _TCNForecastNet:
    def __new__(cls, input_size: int, horizon: int, channels: int, kernel_size: int, dropout: float):
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.network = nn.Sequential(
                    _CausalBlock(input_size, channels, kernel_size, dilation=1, dropout=dropout),
                    _CausalBlock(channels, channels, kernel_size, dilation=2, dropout=dropout),
                    _CausalBlock(channels, channels, kernel_size, dilation=4, dropout=dropout),
                )
                self.head = nn.Linear(channels, horizon)

            def forward(self, x):
                out = self.network(x.transpose(1, 2))
                return self.head(out[:, :, -1])

        return Net()


def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(true) & np.isfinite(pred)
    if not np.any(mask):
        return math.inf
    return float(np.mean((pred[mask] - true[mask]) ** 2))


def _parse_int_choices(value: Any) -> list[int]:
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(v.strip()) for v in str(value).split(",") if v.strip()]
