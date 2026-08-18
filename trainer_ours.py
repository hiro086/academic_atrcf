from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from datasets import DatasetLoader
from model_ours import OursModel


class OursTrainer:
    """Train and evaluate the ARCF-TCN model in an isolated result tree."""

    def __init__(self, config_path: str | Path = "config_ours.yaml") -> None:
        self.loader = DatasetLoader()
        self.framework_dir = self.loader.framework_dir
        self.base_config = self.loader.config
        self.ours_config = self._load_simple_yaml(self.framework_dir / config_path)
        self.results_dir = self.framework_dir / "results_ours"
        self.prediction_dir = self.results_dir / "predictions"
        self.model_dir = self.results_dir / "model_artifacts"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.prediction_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def write_data_availability(self) -> tuple[Path, Path]:
        availability = self.loader.write_data_availability()
        inventory = self.loader.write_dataset_inventory()
        return availability, inventory

    def write_preprocessing_summary(self) -> Path:
        rows = []
        for dataset_name in self.loader.available_datasets():
            cells = self.loader.load_dataset(dataset_name)
            x, y, _ = self.loader.build_windows(cells)
            rows.append(
                {
                    "dataset": dataset_name,
                    "cell_count": len(cells),
                    "window_count": len(x),
                    "x_shape": "x".join(map(str, x.shape)),
                    "y_shape": "x".join(map(str, y.shape)),
                    "window_size": self.base_config["window_size"],
                    "prediction_horizon": self.base_config["prediction_horizon"],
                    "split_strategy": self.base_config["split_strategy"],
                }
            )
        output = self.results_dir / "preprocessing_summary.csv"
        pd.DataFrame(rows).to_csv(output, index=False)
        return output

    def train(self) -> Path:
        status_rows: list[dict[str, Any]] = []
        search_rows: list[dict[str, Any]] = []
        fusion_rows: list[dict[str, Any]] = []
        loss_rows: list[dict[str, Any]] = []
        prediction_path = self.prediction_dir / "predictions.csv"
        with prediction_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._prediction_fieldnames())
            writer.writeheader()
            for dataset_name in self.loader.available_datasets():
                try:
                    cells = self.loader.load_dataset(dataset_name)
                    folds = self.loader.get_folds(cells)
                except Exception as exc:
                    status_rows.append(self._dataset_status(dataset_name, "failed_dataset_load", repr(exc)))
                    continue
                for seed in self.base_config.get("seeds", [42]):
                    for fold in folds:
                        if len(fold["train_x"]) == 0 or len(fold["validation_x"]) == 0 or len(fold["test_x"]) == 0:
                            status_rows.append(self._status(dataset_name, seed, fold, "skipped_empty_fold", "No sequence windows"))
                            continue
                        try:
                            cfg = self._model_config(seed)
                            model = OursModel(cfg)
                            t0 = time.perf_counter()
                            model.fit(fold["train_x"], fold["train_y"], fold["validation_x"], fold["validation_y"])
                            train_time = time.perf_counter() - t0
                            t1 = time.perf_counter()
                            validation_pred = model.predict(fold["validation_x"])
                            test_pred = model.predict(fold["test_x"])
                            inference_time = time.perf_counter() - t1
                            artifact = self.model_dir / f"{dataset_name}_Ours_{seed}_{fold['fold_id']}.pt"
                            model.save(artifact)
                            param_count = model.count_params()
                            self._write_predictions(writer, dataset_name, seed, fold, "validation", validation_pred, train_time, inference_time, param_count)
                            self._write_predictions(writer, dataset_name, seed, fold, "test", test_pred, train_time, inference_time, param_count)
                            for row in model.search_history:
                                search_rows.append({"dataset": dataset_name, "fold_id": fold["fold_id"], "seed": seed, **row})
                            for row in model.fusion_history:
                                fusion_rows.append({"dataset": dataset_name, "fold_id": fold["fold_id"], "seed": seed, **row})
                            for row in model.loss_history:
                                loss_rows.append({"dataset": dataset_name, "fold_id": fold["fold_id"], "seed": seed, **row})
                            detail = f"{artifact}; best_params={model.best_params}; residual_alpha={model.best_weight:.4f}; device={model.device_name}"
                            status_rows.append(self._status(dataset_name, seed, fold, "ok", detail))
                        except Exception as exc:
                            status_rows.append(self._status(dataset_name, seed, fold, "failed", repr(exc)))
        pd.DataFrame(status_rows).to_csv(self.results_dir / "run_status.csv", index=False)
        pd.DataFrame(search_rows).to_csv(self.results_dir / "search_history.csv", index=False)
        pd.DataFrame(fusion_rows).to_csv(self.results_dir / "fusion_history.csv", index=False)
        pd.DataFrame(loss_rows).to_csv(self.results_dir / "loss_history.csv", index=False)
        return prediction_path

    def predict(self) -> Path:
        prediction_path = self.prediction_dir / "predictions.csv"
        if not prediction_path.exists():
            raise FileNotFoundError("Run 03_train_ours.py before 04_predict_ours.py")
        frame = pd.read_csv(prediction_path)
        output = self.results_dir / "prediction_manifest.csv"
        frame.groupby(["task", "dataset", "model", "model_family", "tuned", "seed", "split", "fold_id"], as_index=False).agg(
            prediction_rows=("pred", "count"),
            first_true=("true", "first"),
            first_pred=("pred", "first"),
        ).to_csv(output, index=False)
        return output

    def analyze(self) -> Path:
        prediction_path = self.prediction_dir / "predictions.csv"
        if not prediction_path.exists():
            raise FileNotFoundError("Run 03_train_ours.py before 05_analyze_results_ours.py")
        preds = pd.read_csv(prediction_path)
        group_cols = ["task", "dataset", "model", "model_family", "tuned", "seed", "split"]
        rows = []
        for keys, group in preds.groupby(group_cols, dropna=False):
            row = dict(zip(group_cols, keys))
            row.update(self._evaluate(group["true"], group["pred"]))
            row.update(
                {
                    "training_time_sec": group["train_time_sec"].max(),
                    "inference_time_sec": group["inference_time_sec"].sum(),
                    "param_count": group["param_count"].max(),
                    "status": "ok",
                }
            )
            rows.append(row)
        metrics = pd.DataFrame(rows)
        output = self.results_dir / "metrics.csv"
        metrics.to_csv(output, index=False)
        self._write_figures(preds, metrics)
        return output

    def _model_config(self, seed: int) -> dict[str, Any]:
        return {
            **self.base_config,
            **self.ours_config.get("model", {}),
            **self.ours_config.get("search", {}),
            **self.ours_config.get("training", {}),
            "seed": int(seed),
            "prediction_horizon": int(self.base_config["prediction_horizon"]),
        }

    @staticmethod
    def _evaluate(y_true, y_pred) -> dict[str, float]:
        true = np.asarray(y_true, dtype=float).reshape(-1)
        pred = np.asarray(y_pred, dtype=float).reshape(-1)
        mask = np.isfinite(true) & np.isfinite(pred)
        true = true[mask]
        pred = pred[mask]
        if len(true) == 0:
            return {"RMSE": math.nan, "MAE": math.nan, "MSE": math.nan, "MAPE": math.nan, "R2": math.nan}
        error = pred - true
        mse = float(np.mean(error**2))
        mae = float(np.mean(np.abs(error)))
        denom = np.where(np.abs(true) < 1e-8, np.nan, np.abs(true))
        ss_res = float(np.sum(error**2))
        ss_tot = float(np.sum((true - np.mean(true)) ** 2))
        return {
            "RMSE": float(math.sqrt(mse)),
            "MAE": mae,
            "MSE": mse,
            "MAPE": float(np.nanmean(np.abs(error) / denom) * 100.0),
            "R2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else math.nan,
        }

    def _write_predictions(self, writer, dataset_name: str, seed: int, fold: dict[str, Any], split: str, pred: np.ndarray, train_time: float, inference_time: float, param_count: int) -> None:
        y_true = fold[f"{split}_y"]
        meta = fold[f"{split}_meta"]
        eval_cell = fold["validation_cell"] if split == "validation" else fold["test_cell"]
        for sample_idx in range(len(y_true)):
            for horizon_idx in range(y_true.shape[1]):
                writer.writerow(
                    {
                        "task": self.base_config["task_type"],
                        "dataset": dataset_name,
                        "model": "Ours",
                        "model_family": "ARCF-TCN",
                        "tuned": True,
                        "seed": seed,
                        "split": split,
                        "fold_id": fold["fold_id"],
                        "validation_cell": fold["validation_cell"],
                        "test_cell": fold["test_cell"],
                        "eval_cell": eval_cell,
                        "sample_index": sample_idx,
                        "horizon_index": horizon_idx + 1,
                        "cycle": meta[sample_idx].get("cycle", sample_idx) if sample_idx < len(meta) else sample_idx,
                        "true": float(y_true[sample_idx, horizon_idx]),
                        "pred": float(pred[sample_idx, horizon_idx]),
                        "train_time_sec": train_time,
                        "inference_time_sec": inference_time,
                        "param_count": param_count,
                    }
                )

    @staticmethod
    def _prediction_fieldnames() -> list[str]:
        return [
            "task",
            "dataset",
            "model",
            "model_family",
            "tuned",
            "seed",
            "split",
            "fold_id",
            "validation_cell",
            "test_cell",
            "eval_cell",
            "sample_index",
            "horizon_index",
            "cycle",
            "true",
            "pred",
            "train_time_sec",
            "inference_time_sec",
            "param_count",
        ]

    def _dataset_status(self, dataset: str, status: str, detail: str) -> dict[str, Any]:
        return {"task": self.base_config["task_type"], "dataset": dataset, "model": "Ours", "seed": "", "fold_id": "", "validation_cell": "", "test_cell": "", "status": status, "detail": detail}

    def _status(self, dataset: str, seed: int, fold: dict[str, Any], status: str, detail: str) -> dict[str, Any]:
        return {"task": self.base_config["task_type"], "dataset": dataset, "model": "Ours", "seed": seed, "fold_id": fold["fold_id"], "validation_cell": fold["validation_cell"], "test_cell": fold["test_cell"], "status": status, "detail": detail}

    def _write_figures(self, preds: pd.DataFrame, metrics: pd.DataFrame) -> None:
        figures = self.results_dir / "figures"
        figures.mkdir(parents=True, exist_ok=True)
        _simple_svg(figures / "metric_curve.svg", metrics[metrics["split"].eq("test")], "RMSE")
        loss_path = self.results_dir / "loss_history.csv"
        if loss_path.exists():
            loss = pd.read_csv(loss_path)
            _line_svg(figures / "loss_curve.svg", loss.groupby("epoch", as_index=False)["validation_loss"].mean()["validation_loss"].tolist(), "Validation loss")
        sample = preds[preds["split"].eq("test")].head(600)
        _line_svg(figures / "true_vs_pred.svg", sample["true"].tolist(), "True values")
        _line_svg(figures / "pred_curve.svg", sample["pred"].tolist(), "Predicted values")

    @staticmethod
    def _load_simple_yaml(path: Path) -> dict[str, Any]:
        result: dict[str, Any] = {}
        current: dict[str, Any] | None = None
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.split("#", 1)[0].rstrip()
            if not line:
                continue
            if not line.startswith(" ") and line.endswith(":"):
                current = {}
                result[line[:-1].strip()] = current
            elif current is not None and ":" in line:
                key, value = line.strip().split(":", 1)
                current[key.strip()] = _parse_scalar(value.strip())
        return result


def _parse_scalar(value: str) -> Any:
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _simple_svg(path: Path, frame: pd.DataFrame, metric: str) -> None:
    bars = [(str(r.dataset), float(getattr(r, metric))) for r in frame.itertuples()]
    width, height = 720, 320
    max_v = max([v for _, v in bars], default=1.0) or 1.0
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">', '<rect width="100%" height="100%" fill="white"/>', f'<text x="20" y="30" font-size="18">{metric}</text>']
    for i, (label, value) in enumerate(bars):
        x = 90 + i * 120
        h = (height - 100) * value / max_v
        parts.append(f'<rect x="{x}" y="{height-55-h:.1f}" width="70" height="{h:.1f}" fill="#4F81BD"/>')
        parts.append(f'<text x="{x}" y="{height-30}" font-size="11">{label}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _line_svg(path: Path, values: list[float], title: str) -> None:
    if not values:
        return
    width, height = 720, 320
    lo, hi = min(values), max(values)
    span = hi - lo or 1.0
    pts = []
    for i, value in enumerate(values):
        x = 45 + i * (width - 80) / max(1, len(values) - 1)
        y = height - 45 - (float(value) - lo) * (height - 90) / span
        pts.append(f"{x:.1f},{y:.1f}")
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><rect width="100%" height="100%" fill="white"/><text x="20" y="30" font-size="18">{title}</text><polyline points="{" ".join(pts)}" fill="none" stroke="#C0504D" stroke-width="2"/></svg>', encoding="utf-8")
