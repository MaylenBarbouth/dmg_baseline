# src/logger_utils.py
import json
import logging
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import optuna
import pandas as pd


def setup_loggers(outputs_dir: str) -> logging.Logger:
    """Configura un logger dual (consola limpia + archivo histórico con timestamp)."""
    log_dir = Path(outputs_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exec_log_path = log_dir / f"execution_{timestamp}.log"
    err_log_path = log_dir / "errors.log"

    # Silenciar warnings ruidosos en la terminal
    warnings.filterwarnings("ignore")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    logger = logging.getLogger("DMG_Baseline")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # 1. Handler para archivo completo (DEBUG)
    fh = logging.FileHandler(exec_log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
    )

    # 2. Handler exclusivo para errores (ERROR)
    eh = logging.FileHandler(err_log_path, encoding="utf-8")
    eh.setLevel(logging.ERROR)
    eh.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
    )

    # 3. Handler para consola (INFO, formato limpio)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(eh)
    logger.addHandler(ch)

    return logger


def atomic_write_json(data: Dict[str, Any], filepath: Path) -> None:
    """Escribe un archivo JSON primero en .tmp y luego lo renombra (a prueba de cortes de luz)."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = filepath.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, filepath)


def atomic_write_dataframe(df: pd.DataFrame, filepath: Path) -> None:
    """Escribe un DataFrame (CSV o Parquet) de forma atómica."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    suffix = filepath.suffix.lower()
    tmp_path = filepath.with_suffix(f"{suffix}.tmp")

    if suffix == ".parquet":
        df.to_parquet(tmp_path, index=False)
    elif suffix == ".csv":
        df.to_csv(tmp_path, index=False)
    else:
        raise ValueError(f"Formato de archivo no soportado: {suffix}")

    os.replace(tmp_path, filepath)