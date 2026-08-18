from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class CellSeries:
    """One battery/cell degradation sequence after unified preprocessing."""

    dataset: str
    cell_id: str
    cycle: np.ndarray
    capacity: np.ndarray
    target: np.ndarray
    rated_capacity: float
    eol_threshold: float


class DatasetLoader:
    """Load raw or processed battery series through one benchmark interface."""

    def __init__(self, config_path: str | Path = "configs/benchmark_config.json") -> None:
        self.framework_dir = Path(__file__).resolve().parents[1]
        self.repo_root = self._find_repo_root(self.framework_dir)
        self.config_path = (self.framework_dir / config_path).resolve()
        with self.config_path.open("r", encoding="utf-8") as f:
            self.config = json.load(f)
        self.processed_dir = self.framework_dir / "datasets" / "processed"
        self.metadata_dir = self.framework_dir / "metadata"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _find_repo_root(start: Path) -> Path:
        for parent in [start, *start.parents]:
            if (parent / "datasets").exists() and (parent / "experiments").exists():
                return parent
        return start.parents[4]

    def _resolve_existing(self, candidates: Iterable[str]) -> Path | None:
        for candidate in candidates:
            path = self.repo_root / candidate
            if path.exists():
                return path
        return None

    def available_datasets(self) -> List[str]:
        return [d["name"] for d in self.config["datasets"] if d.get("enabled", True)]

    def load_all(self) -> Dict[str, Dict[str, CellSeries]]:
        return {name: self.load_dataset(name) for name in self.available_datasets()}

    def load_dataset(self, dataset_name: str) -> Dict[str, CellSeries]:
        dataset_cfg = self._dataset_cfg(dataset_name)
        processed_path = self.processed_dir / f"{dataset_name}_{self.config['target']['name']}_cells.csv"
        if processed_path.exists() and self.config.get("reuse_processed", True):
            return self._load_processed_cells(processed_path, dataset_cfg)
        raw_cells = self._load_raw_cells(dataset_cfg)
        self._write_processed_cells(processed_path, raw_cells)
        return raw_cells

    def _dataset_cfg(self, dataset_name: str) -> dict:
        for dataset_cfg in self.config["datasets"]:
            if dataset_cfg["name"] == dataset_name:
                return dataset_cfg
        raise KeyError(f"Dataset is not configured: {dataset_name}")

    def _load_raw_cells(self, dataset_cfg: dict) -> Dict[str, CellSeries]:
        name = dataset_cfg["name"].upper()
        if name == "NASA":
            return self._load_nasa(dataset_cfg)
        if name == "CALCE":
            return self._load_calce(dataset_cfg)
        raise ValueError(f"Unsupported dataset loader: {dataset_cfg['name']}")

    def _load_nasa(self, dataset_cfg: dict) -> Dict[str, CellSeries]:
        path = self._resolve_existing(dataset_cfg["raw_candidates"])
        if path is None:
            raise FileNotFoundError(f"NASA npy file not found from {dataset_cfg['raw_candidates']}")
        data = np.load(path, allow_pickle=True).item()
        cells: Dict[str, CellSeries] = {}
        for cell_id, value in data.items():
            if isinstance(value, (list, tuple)) and len(value) >= 2:
                cycle = np.asarray(value[0], dtype=float)
                capacity = np.asarray(value[1], dtype=float)
            elif isinstance(value, dict):
                cycle = np.asarray(value.get("cycle", np.arange(len(value["capacity"])) + 1), dtype=float)
                capacity = np.asarray(value["capacity"], dtype=float)
            else:
                raise TypeError(f"Unsupported NASA payload for {cell_id}: {type(value)}")
            cells[str(cell_id)] = self._build_cell(dataset_cfg, str(cell_id), cycle, capacity)
        return cells

    def _load_calce(self, dataset_cfg: dict) -> Dict[str, CellSeries]:
        path = self._resolve_existing(dataset_cfg["raw_candidates"])
        if path is None:
            raise FileNotFoundError(f"CALCE npy file not found from {dataset_cfg['raw_candidates']}")
        data = np.load(path, allow_pickle=True).item()
        cells: Dict[str, CellSeries] = {}
        for cell_id, value in data.items():
            if hasattr(value, "columns"):
                cycle = np.asarray(value["cycle"], dtype=float)
                capacity = np.asarray(value["capacity"], dtype=float)
            elif isinstance(value, dict):
                cycle = np.asarray(value.get("cycle", np.arange(len(value["capacity"])) + 1), dtype=float)
                capacity = np.asarray(value["capacity"], dtype=float)
            else:
                raise TypeError(f"Unsupported CALCE payload for {cell_id}: {type(value)}")
            cells[str(cell_id)] = self._build_cell(dataset_cfg, str(cell_id), cycle, capacity)
        return cells

    def _build_cell(self, dataset_cfg: dict, cell_id: str, cycle: np.ndarray, capacity: np.ndarray) -> CellSeries:
        mask = np.isfinite(cycle) & np.isfinite(capacity)
        cycle = cycle[mask]
        capacity = capacity[mask]
        order = np.argsort(cycle)
        cycle = cycle[order]
        capacity = capacity[order]
        rated_capacity = float(dataset_cfg.get("rated_capacity") or np.nanmax(capacity))
        threshold = float(dataset_cfg.get("eol_threshold_ratio", 0.8)) * rated_capacity
        target = self._target_from_capacity(capacity, rated_capacity, threshold)
        return CellSeries(
            dataset=dataset_cfg["name"],
            cell_id=cell_id,
            cycle=cycle.astype(float),
            capacity=capacity.astype(float),
            target=target.astype(float),
            rated_capacity=rated_capacity,
            eol_threshold=threshold,
        )

    def _target_from_capacity(self, capacity: np.ndarray, rated_capacity: float, threshold: float) -> np.ndarray:
        target_name = self.config["target"]["name"].lower()
        if target_name == "rul":
            eol_index = self._first_threshold_crossing(capacity, threshold)
            return np.maximum(eol_index - np.arange(len(capacity)), 0).astype(float)
        if target_name == "soh":
            return capacity / rated_capacity
        if target_name == "capacity":
            return capacity.astype(float)
        raise ValueError(f"Unsupported target: {target_name}")

    @staticmethod
    def _first_threshold_crossing(values: np.ndarray, threshold: float) -> int:
        crossings = np.where(values <= threshold)[0]
        return int(crossings[0]) if len(crossings) else int(len(values) - 1)

    def _write_processed_cells(self, path: Path, cells: Dict[str, CellSeries]) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "dataset",
                    "cell_id",
                    "cycle",
                    "capacity",
                    "target",
                    "rated_capacity",
                    "eol_threshold",
                ],
            )
            writer.writeheader()
            for cell in cells.values():
                for idx in range(len(cell.target)):
                    writer.writerow(
                        {
                            "dataset": cell.dataset,
                            "cell_id": cell.cell_id,
                            "cycle": cell.cycle[idx],
                            "capacity": cell.capacity[idx],
                            "target": cell.target[idx],
                            "rated_capacity": cell.rated_capacity,
                            "eol_threshold": cell.eol_threshold,
                        }
                    )

    def _load_processed_cells(self, path: Path, dataset_cfg: dict) -> Dict[str, CellSeries]:
        frame = pd.read_csv(path)
        cells: Dict[str, CellSeries] = {}
        for cell_id, group in frame.groupby("cell_id", sort=False):
            cells[str(cell_id)] = CellSeries(
                dataset=dataset_cfg["name"],
                cell_id=str(cell_id),
                cycle=group["cycle"].to_numpy(dtype=float),
                capacity=group["capacity"].to_numpy(dtype=float),
                target=group["target"].to_numpy(dtype=float),
                rated_capacity=float(group["rated_capacity"].iloc[0]),
                eol_threshold=float(group["eol_threshold"].iloc[0]),
            )
        return cells

    def build_windows(self, cells: Dict[str, CellSeries]) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
        window_size = int(self.config["window_size"])
        horizon = int(self.config["prediction_horizon"])
        xs: List[np.ndarray] = []
        ys: List[np.ndarray] = []
        meta: List[dict] = []
        for cell in cells.values():
            values = cell.target.astype(float)
            for start in range(0, len(values) - window_size - horizon + 1):
                x = values[start : start + window_size]
                y = values[start + window_size : start + window_size + horizon]
                xs.append(x.reshape(window_size, 1))
                ys.append(y.reshape(horizon))
                meta.append(
                    {
                        "dataset": cell.dataset,
                        "cell_id": cell.cell_id,
                        "start_index": start,
                        "cycle": float(cell.cycle[start + window_size - 1]),
                        "rated_capacity": cell.rated_capacity,
                        "eol_threshold": cell.eol_threshold,
                    }
                )
        if not xs:
            return np.empty((0, window_size, 1)), np.empty((0, horizon)), meta
        return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), meta

    def leave_one_cell_out(self, cells: Dict[str, CellSeries]) -> List[dict]:
        folds = []
        cell_ids = sorted(cells)
        for idx, test_cell in enumerate(cell_ids):
            validation_cell = cell_ids[(idx + 1) % len(cell_ids)] if len(cell_ids) > 1 else test_cell
            train_cells = {k: v for k, v in cells.items() if k not in {test_cell, validation_cell}}
            if not train_cells:
                train_cells = {k: v for k, v in cells.items() if k != test_cell}
            validation_cells = {validation_cell: cells[validation_cell]}
            test_cells = {test_cell: cells[test_cell]}
            train_x, train_y, train_meta = self.build_windows(train_cells)
            validation_x, validation_y, validation_meta = self.build_windows(validation_cells)
            test_x, test_y, test_meta = self.build_windows(test_cells)
            folds.append(
                {
                    "fold_id": f"test_{test_cell}",
                    "validation_cell": validation_cell,
                    "test_cell": test_cell,
                    "train_x": train_x,
                    "train_y": train_y,
                    "train_meta": train_meta,
                    "validation_x": validation_x,
                    "validation_y": validation_y,
                    "validation_meta": validation_meta,
                    "test_x": test_x,
                    "test_y": test_y,
                    "test_meta": test_meta,
                }
            )
        return folds

    def chronological_split(self, cells: Dict[str, CellSeries]) -> List[dict]:
        folds = []
        ratio = float(self.config.get("train_ratio", 0.8))
        for cell_id, cell in sorted(cells.items()):
            train_end = max(int(len(cell.target) * ratio), int(self.config["window_size"]) + 1)
            train_cells = {
                cell_id: CellSeries(
                    cell.dataset,
                    cell.cell_id,
                    cell.cycle[:train_end],
                    cell.capacity[:train_end],
                    cell.target[:train_end],
                    cell.rated_capacity,
                    cell.eol_threshold,
                )
            }
            test_cells = {
                cell_id: CellSeries(
                    cell.dataset,
                    cell.cell_id,
                    cell.cycle[train_end - int(self.config["window_size"]) :],
                    cell.capacity[train_end - int(self.config["window_size"]) :],
                    cell.target[train_end - int(self.config["window_size"]) :],
                    cell.rated_capacity,
                    cell.eol_threshold,
                )
            }
            train_x, train_y, train_meta = self.build_windows(train_cells)
            test_x, test_y, test_meta = self.build_windows(test_cells)
            folds.append(
                {
                    "fold_id": f"chronological_{cell_id}",
                    "validation_cell": cell_id,
                    "test_cell": cell_id,
                    "train_x": train_x,
                    "train_y": train_y,
                    "train_meta": train_meta,
                    "validation_x": test_x,
                    "validation_y": test_y,
                    "validation_meta": test_meta,
                    "test_x": test_x,
                    "test_y": test_y,
                    "test_meta": test_meta,
                }
            )
        return folds

    def get_folds(self, cells: Dict[str, CellSeries]) -> List[dict]:
        split = self.config.get("split_strategy", "leave_one_cell_out")
        if split == "leave_one_cell_out":
            return self.leave_one_cell_out(cells)
        if split == "chronological_80_20":
            return self.chronological_split(cells)
        raise ValueError(f"Unsupported split strategy: {split}")

    def write_dataset_inventory(self) -> Path:
        rows = []
        for dataset_name in self.available_datasets():
            cells = self.load_dataset(dataset_name)
            x, y, _ = self.build_windows(cells)
            for cell in cells.values():
                rows.append(
                    {
                        "task": self.config["task_type"],
                        "dataset": dataset_name,
                        "cell_id": cell.cell_id,
                        "sample_count": len(cell.target),
                        "feature_count": len(self.config["input_features"]),
                        "time_window_length": self.config["window_size"],
                        "window_size": self.config["window_size"],
                        "prediction_horizon": self.config["prediction_horizon"],
                        "label_definition": self.config["target"]["label_definition"],
                        "rated_capacity": cell.rated_capacity,
                        "eol_threshold": cell.eol_threshold,
                        "window_sample_count": len(x),
                        "target_min": float(np.nanmin(cell.target)),
                        "target_max": float(np.nanmax(cell.target)),
                    }
                )
        output = self.metadata_dir / "dataset_inventory.csv"
        pd.DataFrame(rows).to_csv(output, index=False)
        return output

    def write_data_availability(self) -> Path:
        rows = []
        for dataset_cfg in self.config["datasets"]:
            path = self._resolve_existing(dataset_cfg["raw_candidates"])
            rows.append(
                {
                    "task": self.config["task_type"],
                    "dataset": dataset_cfg["name"],
                    "available": path is not None,
                    "resolved_path": "" if path is None else str(path),
                    "raw_candidates": "|".join(dataset_cfg["raw_candidates"]),
                }
            )
        output = self.metadata_dir / "data_availability.csv"
        pd.DataFrame(rows).to_csv(output, index=False)
        return output
