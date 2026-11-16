# 07_diagnosticar_logreg_colapso.py
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def main():
    print("=" * 80)
    print(" DIAGNÓSTICO DE COLAPSO: REGRESIÓN LOGÍSTICA EN SET 3 (LABORATORIO FULL) ")
    print("=" * 80)

    # 1. Cargar configuración y datos
    with open("config/experiment_config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    proc_dir = Path(config["paths"]["processed_dir"])
    df = pd.read_parquet(proc_dir / "cohort_A_general.parquet")
    splits = pd.read_parquet(proc_dir / "splits_cv_5x5.parquet")

    # 2. Obtener columnas del Set 3 (excluyendo IMC y TALLA como dicta la LogReg)
    cols_full = config["columns"]["feature_sets"]["set_3_full"]
    high_null = config["columns"]["high_null_drop_in_logreg"]
    active_cols = [c for c in cols_full if c not in high_null]
    
    target_col = config["schema"]["target_col"]
    preg_col = config["schema"]["pregnancy_id"]

    # 3. Separar Train y Test para el Repeat 0, Fold 0
    mask_split = (splits["cohort"] == "A_all") & (splits["repeat_idx"] == 0) & (splits["fold_idx"] == 0)
    test_ids = set(splits.loc[mask_split, preg_col])
    
    df_train = df[~df[preg_col].isin(test_ids)].copy()
    df_test = df[df[preg_col].isin(test_ids)].copy()

    X_tr = df_train[active_cols]
    y_tr = df_train[target_col].values.astype(int)
    X_te = df_test[active_cols]

    print(f"[*] Datos cargados: {len(X_tr)} filas Train | {len(active_cols)} columnas activas.")

    # -------------------------------------------------------------------------
    # CHEQUEO 1: INFINITOS Y OUTLIERS EN DATA CRUDA
    # -------------------------------------------------------------------------
    print("\n--- CHEQUEO 1: INFINITOS EN DATOS CRUDOS ---")
    inf_counts = np.isinf(X_tr.select_dtypes('number')).sum()
    inf_cols = inf_counts[inf_counts > 0]
    if len(inf_cols) > 0:
        print("⚠️ ALERTA: Se encontraron valores infinitos en:")
        print(inf_cols.to_string())
    else:
        print("✔ No hay valores infinitos en el set de entrenamiento crudo.")

    # -------------------------------------------------------------------------
    # CONSTRUIR Y ENTRENAR EL PIPELINE EXACTO
    # -------------------------------------------------------------------------
    # Usamos un C=1.0 estandar (ya sabemos que colapsaba con Optuna, probemos con la regularización por defecto)
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
        ('scaler', RobustScaler()),
        ('clf', LogisticRegression(
            penalty='elasticnet', solver='saga', l1_ratio=0.5, C=1.0, 
            class_weight=None, max_iter=2500, random_state=42
        ))
    ])
    
    print("\n[*] Entrenando pipeline (puede tardar unos segundos si el solver le cuesta converger)...")
    pipe.fit(X_tr, y_tr)
    
    clf = pipe.named_steps['clf']
    scaler = pipe.named_steps['scaler']
    imputer = pipe.named_steps['imputer']
    
    X_tr_imputed = imputer.transform(X_tr)
    X_tr_scaled = scaler.transform(X_tr_imputed)
    feature_names = imputer.get_feature_names_out(active_cols)

    # -------------------------------------------------------------------------
    # CHEQUEO 2: CONVERGENCIA DEL SOLVER
    # -------------------------------------------------------------------------
    print("\n--- CHEQUEO 2: CONVERGENCIA DEL ALGORITMO SAGA ---")
    print(f"Iteraciones consumidas: {clf.n_iter_[0]} / Máximo permitido: {clf.max_iter}")
    if clf.n_iter_[0] >= clf.max_iter:
        print("⚠️ ALERTA: El algoritmo 'saga' se cortó por falta de iteraciones. Superficie de error muy plana (colinealidad severa).")
    else:
        print("✔ El algoritmo convergió exitosamente.")

    # -------------------------------------------------------------------------
    # CHEQUEO 3: EXPLOSIÓN DEL ESCALADOR (EL TALÓN DE AQUILES DEL ROBUSTSCALER)
    # -------------------------------------------------------------------------
    print("\n--- CHEQUEO 3: VALORES EXTREMOS POST-ESCALADO ---")
    # El RobustScaler escala dividiendo por el Rango Intercuartílico (IQR). 
    # Si una variable tiene el percentil 25 igual al percentil 75, el IQR es casi 0, 
    # y al dividir, los valores se disparan a miles o millones.
    max_vals = np.abs(X_tr_scaled).max(axis=0)
    df_max = pd.DataFrame({'Feature': feature_names, 'Max_Abs_Scaled': max_vals})
    df_max = df_max.sort_values('Max_Abs_Scaled', ascending=False)
    
    print("Top 5 variables con valores absolutos más gigantescos DESPUÉS de escalar:")
    print(df_max.head(5).to_string(index=False))
    
    if df_max['Max_Abs_Scaled'].iloc[0] > 100:
        print("⚠️ ALERTA: Tenés variables con escalas > 100. El solver logístico pierde precisión numérica y colapsa.")

    # -------------------------------------------------------------------------
    # CHEQUEO 4: PESOS DEL MODELO (COEFICIENTES) E INTERCEPTO
    # -------------------------------------------------------------------------
    print("\n--- CHEQUEO 4: ESTADO DE LOS COEFICIENTES (PESOS) ---")
    ceros = np.sum(clf.coef_ == 0)
    totales = clf.coef_.size
    print(f"Coeficientes apagados exactos (a cero): {ceros} de {totales}")
    print(f"Intercepto base del modelo: {clf.intercept_[0]:.4f}")
    
    # -------------------------------------------------------------------------
    # CHEQUEO 5: DISTRIBUCIÓN DE LAS PROBABILIDADES EN TEST
    # -------------------------------------------------------------------------
    print("\n--- CHEQUEO 5: PROBABILIDADES PREDICHAS EN TEST ---")
    p = pipe.predict_proba(X_te)[:, 1]
    print(f"Probabilidades -> Mínima: {p.min():.4f} | Mediana: {np.median(p):.4f} | Máxima: {p.max():.4f}")
    print(f"Log-Loss crudo: {-np.mean(y_tr)*np.log(0.5) - (1-np.mean(y_tr))*np.log(0.5):.4f} (Si predijera todo 50%)")
    
    print("\n=======================================================================")
    print(" CONCLUSIÓN DEL DIAGNÓSTICO:")
    if np.median(p) > 0.4 and np.median(p) < 0.6:
        print(" ❌ CONFIRMADO: El modelo colapsó a predecir ~50%.")
        print(" 👉 Mirá el Chequeo 3 y el Chequeo 2 para encontrar al culpable.")
    else:
        print(" ✔ El modelo no colapsó con esta regularización base.")
    print("=======================================================================")

if __name__ == "__main__":
    main()