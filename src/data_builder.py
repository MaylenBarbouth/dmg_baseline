# src/data_builder.py
import hashlib
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def compute_file_sha256(filepath: Path) -> str:
    """Calcula el hash SHA256 del archivo crudo para trazabilidad."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def apply_deterministic_preprocessing(
    df: pd.DataFrame, config: Dict
) -> pd.DataFrame:
    """
    Aplica transformaciones determinísticas a nivel fila (sin data leakage):
    1. Parseo de fecha_1T.
    2. Creación de OBST_IMPUTED_FLAG.
    3. Imputación de primigestas (GRAVIDA=1, PARTOS=0, CESAREAS=0, ABORTOS=0).
    4. Creación de PRIMIGESTA_FLAG para distinguir nulíparas de multíparas sin DMG previa.
    """
    df = df.copy()
    date_col = config["schema"]["date_col"]

    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    else:
        raise KeyError(
            f"No se encontró la columna de fecha '{date_col}' en el dataset."
        )

    obst_zero = config["columns"]["obstetric_zero_impute"]
    obst_one = config["columns"]["obstetric_one_impute"]
    all_obst = obst_one + obst_zero

    # Flag de imputación del bloque obstétrico (1 si alguna de las 4 estaba nula)
    df["OBST_IMPUTED_FLAG"] = df[all_obst].isna().any(axis=1).astype(int)

    # Regla clínica acordada: asumir primigesta en nulos obstétricos
    for col in obst_one:
        df[col] = df[col].fillna(1.0)
    for col in obst_zero:
        df[col] = df[col].fillna(0.0)

    # Flag explícito de primigesta (ayuda a la Regresión Logística con ANTC_DMG=0)
    df["PRIMIGESTA_FLAG"] = (
        (df["GRAVIDA"] <= 1)
        & (df["PARTOS"] == 0)
        & (df["CESAREAS"] == 0)
        & (df["ABORTOS"] == 0)
    ).astype(int)

    return df


def build_cohorts(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Separa Cohorte A (General) y Cohorte B (Sin antecedente de DMG)."""
    cohort_a = df.copy().reset_index(drop=True)
    cohort_b = df[df["ANTC_DMG"] == 0].copy().reset_index(drop=True)
    return cohort_a, cohort_b


def generate_cv_splits(
    df: pd.DataFrame, cohort_name: str, config: Dict
) -> pd.DataFrame:
    """
    Genera 5 repeticiones de un 5-Fold StratifiedGroupKFold agrupando por id_paciente.
    Devuelve una tabla con las asignaciones exactas de test para cada repeat y fold.
    """
    preg_col = config["schema"]["pregnancy_id"]
    pat_col = config["schema"]["patient_id"]
    target_col = config["schema"]["target_col"]

    n_splits = config["validation"]["cv_n_splits"]
    n_repeats = config["validation"]["cv_n_repeats"]
    base_seed = config["validation"]["cv_base_seed"]

    records = []
    for rep in range(n_repeats):
        seed = base_seed + rep * 100
        sgkf = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
        for fold_idx, (train_idx, test_idx) in enumerate(
            sgkf.split(df, df[target_col], df[pat_col])
        ):
            # Verificación estricta de cero fuga de pacientes
            train_pats = set(df.iloc[train_idx][pat_col])
            test_pats = set(df.iloc[test_idx][pat_col])
            assert len(train_pats.intersection(test_pats)) == 0, (
                f"Fuga detectada en {cohort_name} rep={rep} fold={fold_idx}"
            )

            fold_df = pd.DataFrame(
                {
                    "cohort": cohort_name,
                    "repeat_idx": rep,
                    "fold_idx": fold_idx,
                    "split_seed": seed,
                    preg_col: df.iloc[test_idx][preg_col].values,
                    "role": "test",
                }
            )
            records.append(fold_df)

    return pd.concat(records, ignore_index=True)


def generate_temporal_split(
    df: pd.DataFrame, cohort_name: str, config: Dict
) -> pd.DataFrame:
    """
    Ordena por fecha_1T, toma el primer 80% como Train y el último 20% como Test,
    excluyendo de Test cualquier id_paciente que ya haya tenido un embarazo en Train.
    """
    preg_col = config["schema"]["pregnancy_id"]
    pat_col = config["schema"]["patient_id"]
    date_col = config["schema"]["date_col"]
    q_cut = config["validation"]["temporal_train_quantile"]

    df_valid_date = df.dropna(subset=[date_col]).sort_values(
        by=[date_col, preg_col]
    ).reset_index(drop=True)

    cutoff_idx = int(len(df_valid_date) * q_cut)
    train_part = df_valid_date.iloc[:cutoff_idx]
    test_part_raw = df_valid_date.iloc[cutoff_idx:]

    train_patients = set(train_part[pat_col].unique())
    # Purgar de Test temporal las pacientes que ya aparecieron en el pasado (Train)
    test_part_clean = test_part_raw[
        ~test_part_raw[pat_col].isin(train_patients)
    ]

    train_records = pd.DataFrame(
        {
            "cohort": cohort_name,
            preg_col: train_part[preg_col].values,
            "role": "train",
            "Purged_By_Group": False,
        }
    )
    test_records = pd.DataFrame(
        {
            "cohort": cohort_name,
            preg_col: test_part_clean[preg_col].values,
            "role": "test",
            "Purged_By_Group": False,
        }
    )

    return pd.concat([train_records, test_records], ignore_index=True)