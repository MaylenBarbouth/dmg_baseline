# 09_reset_logreg_runs.py
import shutil
from pathlib import Path
import optuna
import yaml


def main() -> None:
    with open("config/experiment_config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    outputs_dir = Path(config["paths"]["outputs_dir"])
    shards_dir = outputs_dir / "shards"
    storage_url = config["paths"]["optuna_db"]

    # 1. Resguardar tablas consolidadas y análisis previos como evidencia metodológica
    backup_dir = outputs_dir / "backup_v1_sin_clipper"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for folder_name in ["consolidated", "analysis"]:
        src_folder = outputs_dir / folder_name
        if src_folder.exists():
            dst_folder = backup_dir / folder_name
            if dst_folder.exists():
                shutil.rmtree(dst_folder)
            shutil.copytree(src_folder, dst_folder)
    print(f"✔ Backup de resultados previos guardado en: {backup_dir}")

    # 2. Eliminar únicamente los shards de Regresión Logística (*__logreg__*)
    removed_shards = 0
    for subfolder in ["metrics", "predictions", "importances", "trials"]:
        sub_path = shards_dir / subfolder
        if not sub_path.exists():
            continue
        for fpath in sub_path.glob("*__logreg__*"):
            fpath.unlink()
            removed_shards += 1

    remaining_lgbm = len(list((shards_dir / "metrics").glob("*__lgbm__*.json")))
    print(f"✔ Shards de LogReg eliminados: {removed_shards} archivos.")
    print(f"✔ Shards de LightGBM intactos: {remaining_lgbm} corridas completas (Esperadas: 560).")

    # 3. Eliminar los estudios de LogReg dentro de la base SQLite de Optuna
    summaries = optuna.study.get_all_study_summaries(storage=storage_url)
    deleted_studies = 0
    for s in summaries:
        if "__logreg__" in s.study_name:
            optuna.delete_study(study_name=s.study_name, storage=storage_url)
            deleted_studies += 1

    print(f"✔ Estudios de Optuna de LogReg reiniciados en SQLite: {deleted_studies}.")
    print("=" * 75)
    print("Listo! Ahora ejecutá: python3 02_run_experiments.py")
    print("Salteará las 560 de LightGBM en 1 segundo y correrá solo las 560 de LogReg.")
    print("=" * 75)


if __name__ == "__main__":
    main()