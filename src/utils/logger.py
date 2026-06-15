"""
MetricLogger — writes training metrics to logs/{run_dir}/.

Layout written per run:
    logs/{run_dir}/
    ├── config.json       config snapshot saved at training start
    ├── metrics.jsonl     one JSON line per log_metrics() call
    ├── train.log         human-readable timestamped text log
    └── metrics.csv       exported at end of training
"""
import csv
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, Optional


class MetricLogger:

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self._jsonl_path = os.path.join(log_dir, "metrics.jsonl")
        self._txt_path   = os.path.join(log_dir, "train.log")
        self._rows: list[dict] = []          # in-memory cache for CSV export

        self._txt = open(self._txt_path, "a", buffering=1)
        self._write_txt(f"{'='*60}")
        self._write_txt(f"Run started: {datetime.now().isoformat()}")
        self._write_txt(f"Log dir:     {os.path.abspath(log_dir)}")
        self._write_txt(f"{'='*60}")

        # Optional TensorBoard
        self._tb = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._tb = SummaryWriter(log_dir=log_dir)
        except ImportError:
            pass

    # ── Public API ────────────────────────────────────────────────────────────

    def log_config(self, config: Any) -> None:
        """Serialize and save the run config as JSON."""
        path = os.path.join(self.log_dir, "config.json")
        try:
            import dataclasses
            raw = dataclasses.asdict(config) if dataclasses.is_dataclass(config) else vars(config)
        except TypeError:
            raw = str(config)
        with open(path, "w") as f:
            json.dump(raw, f, indent=2, default=str)
        self._write_txt(f"Config saved → {path}")

    def log_metrics(self, step: int, epoch: Optional[int] = None, **metrics: float) -> None:
        """
        Log a set of scalar metrics.

        Example:
            logger.log_metrics(step=100, epoch=1, train_loss=0.42, val_auc=0.85)
        """
        row: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "step":      step,
        }
        if epoch is not None:
            row["epoch"] = epoch
        row.update({k: round(float(v), 6) for k, v in metrics.items()})

        # JSONL
        with open(self._jsonl_path, "a") as f:
            f.write(json.dumps(row) + "\n")

        # human-readable
        metric_str = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, float))
        tag = f"[step {step:>6}" + (f" epoch {epoch}" if epoch is not None else "") + f"]  {metric_str}"
        self._write_txt(tag)

        # TensorBoard
        if self._tb is not None:
            for k, v in metrics.items():
                try:
                    self._tb.add_scalar(k, float(v), global_step=step)
                except Exception:
                    pass

        self._rows.append(row)

    def export_csv(self) -> str:
        """Write all logged metrics to metrics.csv. Returns path."""
        path = os.path.join(self.log_dir, "metrics.csv")
        if not self._rows:
            # Fall back: parse the JSONL file
            if os.path.exists(self._jsonl_path):
                with open(self._jsonl_path) as f:
                    self._rows = [json.loads(line) for line in f if line.strip()]

        if not self._rows:
            return path

        fieldnames = list(dict.fromkeys(k for row in self._rows for k in row))
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self._rows)

        self._write_txt(f"Metrics exported → {path}")
        return path

    def close(self) -> None:
        self._write_txt(f"Run ended: {datetime.now().isoformat()}")
        self._txt.close()
        if self._tb is not None:
            self._tb.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _write_txt(self, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        self._txt.write(line + "\n")
