# 01_prepare_data_and_splits.py
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from src.data_builder import (
    apply_deterministic_preprocessing,
    build_cohorts,
    compute_file_sha256,
    generate_cv_splits,
    generate_temporal_split,
)
from src.logger_utils import (
    atomic_write_dataframe,
    atomic_write_json,
    setup_loggers,
)


def main() -> None:
    with open("config/experiment_config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger = setup_loggers(config["paths"]["outputs_dir"])
    raw_path = Path(config["paths"]["raw_data"])
    proc_dir = Path(config["paths"]["processed_dir"])
    proc_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("ETAPA 01: PREPARACIÓN DETERMINÍSTICA Y PARTICIONES MAESTRAS")
    logger.info("=" * 80)

    if not raw_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de entrada en: {raw_path}"
        )

    file_hash = compute_file_sha256(raw_path)
    df_raw = pd.read_csv(raw_path)
    logger.info(
        f"✔ Dataset leído: {len(df_raw)} filas | {df_raw.shape[1]} columnas | SHA256: {file_hash[:12]}..."
    )

    df_clean = apply_deterministic_preprocessing(df_raw, config)
    cohort_a, cohort_b = build_cohorts(df_clean)

    logger.info(
        f"✔ Cohorte A (General): {len(cohort_a)} embarazos ({cohort_a['id_paciente'].nunique()} pacientes, DMG+: {int(cohort_a['DMG'].sum())})"
    )
    logger.info(
        f"✔ Cohorte B (Sin ANTC_DMG): {len(cohort_b)} embarazos ({cohort_b['id_paciente'].nunique()} pacientes, DMG+: {int(cohort_b['DMG'].sum())})"
    )

    # Guardar bases procesadas congeladas
    atomic_write_dataframe(cohort_a, proc_dir / "cohort_A_general.parquet")
    atomic_write_dataframe(cohort_b, proc_dir / "cohort_B_sin_antc.parquet")

    # Generar y guardar splits CV 5x5 agrupados por paciente
    cv_a = generate_cv_splits(cohort_a, "A_all", config)
    cv_b = generate_cv_splits(cohort_b, "B_no_antc", config)
    splits_cv = pd.concat([cv_a, cv_b], ignore_index=True)
    atomic_write_dataframe(splits_cv, proc_dir / "splits_cv_5x5.parquet")
    logger.info(
        "✔ Particiones RepeatedStratifiedGroupKFold (5x5) generadas con 0 fugas de id_paciente."
    )

    # Generar y guardar splits temporales (Out-of-Time)
    temp_a = generate_temporal_split(cohort_a, "A_all", config)
    temp_b = generate_temporal_split(cohort_b, "B_no_antc", config)
    splits_temp = pd.concat([temp_a, temp_b], ignore_index=True)
    atomic_write_dataframe(splits_temp, proc_dir / "splits_temporal.parquet")
    logger.info("✔ Particiones Temporales (80/20 purgadas por paciente) generadas.")

    # Guardar reporte de auditoría
    audit_report = {
        "timestamp": datetime.now().isoformat(),
        "raw_file_path": str(raw_path),
        "raw_file_sha256": file_hash,
        "cohort_A_rows": len(cohort_a),
        "cohort_A_positives": int(cohort_a["DMG"].sum()),
        "cohort_B_rows": len(cohort_b),
        "cohort_B_positives": int(cohort_b["DMG"].sum()),
        "obstetric_imputed_count_A": int(cohort_a["OBST_IMPUTED_FLAG"].sum()),
    }
    atomic_write_json(audit_report, proc_dir / "data_audit_report.json")
    logger.info("✔ Todo listo en data/processed/. Podés ejecutar 02_run_experiments.py")


if __name__ == "__main__":
    main()