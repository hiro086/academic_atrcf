from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from datasets import DatasetLoader
from models import build_model


class BenchmarkTrainer:
    """Run model x dataset folds under one split, feature and metric protocol."""

    def __init__(self, config_path: str | Path = "configs/benchmark_config.json") -> None:
        self.loader = DatasetLoader(config_path)
        self.framework_dir = self.loader.framework_dir
        self.results_dir = self.framework_dir / "results"
        self.prediction_dir = self.results_dir / "predictions"
        self.model_dir = self.results_dir / "model_artifacts"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.prediction_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.config = self.loader.config

    def train(self, selected_models: List[str] | None = None) -> Path:
        status_rows = []
        prediction_path = self.prediction_dir / "predictions.csv"
        with prediction_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
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
                ],
            )
            writer.writeheader()
            for dataset_name in self.loader.available_datasets():
                cells = self.loader.load_dataset(dataset_name)
                folds = self.loader.get_folds(cells)
                for model_cfg in self.config["models"]:
                    if not model_cfg.get("enabled", True):
                        continue
                    if selected_models and model_cfg["name"] not in selected_models:
                        continue
                    for seed in self.config.get("seeds", [42]):
                        cfg = {**self.config, **model_cfg, "seed": seed}
                        model_name = model_cfg["name"]
                        for fold in folds:
                            if len(fold["train_x"]) == 0 or len(fold["test_x"]) == 0:
                                status_rows.append(self._status(dataset_name, model_name, seed, fold, "skipped_empty_fold", "No sequence windows"))
                                continue
                            try:
                                model = build_model(cfg)
                                t0 = time.perf_counter()
                                model.fit(fold["train_x"], fold["train_y"])
                                train_time = time.perf_counter() - t0
                                t1 = time.perf_counter()
                                validation_pred = model.predict(fold["validation_x"])
                                test_pred = model.predict(fold["test_x"])
                                inference_time = time.perf_counter() - t1
                                param_count = model.count_params()
                                artifact = self.model_dir / f"{dataset_name}_{model_name}_{seed}_{fold['fold_id']}.pkl"
                                try:
                                    model.save(artifact)
                                except Exception:
                                    artifact = ""
                                self._write_predictions(writer, dataset_name, model_cfg, seed, fold, "validation", validation_pred, train_time, inference_time, param_count)
                                self._write_predictions(writer, dataset_name, model_cfg, seed, fold, "test", test_pred, train_time, inference_time, param_count)
                                status_rows.append(self._status(dataset_name, model_name, seed, fold, "ok", str(artifact)))
                            except Exception as exc:
                                status = "pending_dependency" if "required" in str(exc).lower() else "failed"
                                status_rows.append(self._status(dataset_name, model_name, seed, fold, status, str(exc)))
        pd.DataFrame(status_rows).to_csv(self.results_dir / "run_status.csv", index=False)
        return prediction_path

    def predict(self) -> Path:
        prediction_path = self.prediction_dir / "predictions.csv"
        if not prediction_path.exists():
            raise FileNotFoundError("Run 03_train.py before 04_predict.py")
        frame = pd.read_csv(prediction_path)
        summary = (
            frame.groupby(["task", "dataset", "model", "seed", "split", "fold_id"], as_index=False)
            .agg(prediction_rows=("pred", "count"), first_true=("true", "first"), first_pred=("pred", "first"))
        )
        output = self.results_dir / "prediction_manifest.csv"
        summary.to_csv(output, index=False)
        return output

    def _status(self, dataset: str, model: str, seed: int, fold: dict, status: str, detail: str) -> dict:
        return {
            "task": self.config["task_type"],
            "dataset": dataset,
            "model": model,
            "seed": seed,
            "fold_id": fold["fold_id"],
            "validation_cell": fold["validation_cell"],
            "test_cell": fold["test_cell"],
            "status": status,
            "detail": detail,
        }

    def _write_predictions(self, writer, dataset_name: str, model_cfg: dict, seed: int, fold: dict, split: str, pred: np.ndarray, train_time: float, inference_time: float, param_count: int) -> None:
        y_true = fold[f"{split}_y"]
        meta = fold[f"{split}_meta"]
        eval_cell = fold[f"{split}_cell"] if f"{split}_cell" in fold else fold["test_cell"]
        for sample_idx in range(len(y_true)):
            for horizon_idx in range(y_true.shape[1]):
                writer.writerow(
                    {
                        "task": self.config["task_type"],
                        "dataset": dataset_name,
                        "model": model_cfg["name"],
                        "model_family": model_cfg.get("model_family", model_cfg["name"]),
                        "tuned": bool(model_cfg.get("tuned", False)),
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
