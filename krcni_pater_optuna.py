import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import optuna
import os
import joblib

# Vytvoření složky pro modely
os.makedirs("models_optuna", exist_ok=True)

# Načti dataset
df = pd.read_csv("data-recovery.csv")

regressors = {}
feature_names_per_kp = {}

# Optimalizace pro každý klíčový bod (kp)
for kp in range(23):
    print(f"Zpracovávám kp{kp}...")
    model_path = f"models_optuna/kp{kp}.pkl"

    # Vytvoření seznamu featur
    features = []
    for i in range(5):
        features += [
            f"pred_kp{kp}_val{i}",
            f"pred_kp{kp}_pos{i}_x",
            f"pred_kp{kp}_pos{i}_y"
        ]
    features += [
        f"pred_kp{kp}_centroid_x",
        f"pred_kp{kp}_centroid_y",
        f"pred_kp{kp}_sigma_x",
        f"pred_kp{kp}_sigma_y"
    ]

    feature_names_per_kp[kp] = features

    X = df[features]
    y = df[[f"target_kp{kp}_x", f"target_kp{kp}_y"]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Pokud model existuje, načteme ho
    if os.path.exists(model_path):
        print(f"Model kp{kp} nalezen, načítám...")
        best_model = joblib.load(model_path)
    else:
        print(f"Optimalizuji kp{kp}...")
        def objective(trial):
            n_estimators = trial.suggest_int('n_estimators', 50, 200)
            max_depth = trial.suggest_int('max_depth', 5, 30)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)

            rf = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42,
                n_jobs=-1
            )
            model = MultiOutputRegressor(rf)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            return mean_squared_error(y_test, y_pred)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=30, show_progress_bar=True)

        print(f"Nejlepší MSE: {study.best_value:.4f}")
        print(f"Parametry: {study.best_params}")

        best_rf = RandomForestRegressor(
            **study.best_params,
            random_state=42,
            n_jobs=-1
        )
        best_model = MultiOutputRegressor(best_rf)
        best_model.fit(X_train, y_train)

        # Uložení modelu
        joblib.dump(best_model, model_path)
        print(f"Model kp{kp} uložen do {model_path}")

    regressors[f"kp{kp}"] = best_model


rf_mse_total = []
nn_mse_total = []

for kp in range(23):
    feature_names = feature_names_per_kp[kp]
    model = regressors[f"kp{kp}"]

    # Připrav data
    X = df[feature_names]
    y_true = df[[f"target_kp{kp}_x", f"target_kp{kp}_y"]]

    # Predikce regrese
    y_pred_rf = model.predict(X)

    # "Predikce" neuronky (vezmeme pos0 přímo z dat)
    y_pred_nn = df[[f"pred_kp{kp}_pos0_x", f"pred_kp{kp}_pos0_y"]].values

    # MSE
    mse_rf = mean_squared_error(y_true, y_pred_rf)
    mse_nn = mean_squared_error(y_true, y_pred_nn)

    rf_mse_total.append(mse_rf)
    nn_mse_total.append(mse_nn)

# Průměrná MSE napříč všemi body
avg_rf_mse = np.mean(rf_mse_total)
avg_nn_mse = np.mean(nn_mse_total)

print("===== Agregované výsledky (celý dataset) =====")
print(f"Průměrná MSE regrese:   {avg_rf_mse:.3f}")
print(f"Průměrná MSE neuronky:  {avg_nn_mse:.3f}")

if avg_rf_mse < avg_nn_mse:
    print("Regrese je přesnější podle MSE!")
else:
    print("Neuronka (pos0) je přesnější podle MSE.")

# Předpoklad: train_test_split je stejný jako dříve (s random_state=42)

# Rozdělení dat pro testovací indexy
_, test_idx = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)

vysledky_df = pd.DataFrame(index=test_idx)

for kp in range(23):
    prefix = f"kp{kp}"
    feature_names = feature_names_per_kp[kp]
    model = regressors[prefix]

    # Data
    X = df[feature_names].iloc[test_idx]
    y_true = df[[f"target_{prefix}_x", f"target_{prefix}_y"]].iloc[test_idx]

    # Predikce regrese
    y_pred_rf = model.predict(X)

    # Predikce založená na pozici s nejvyšší vahou (pos0)
    y_pred_nn = df[[f"pred_{prefix}_pos0_x", f"pred_{prefix}_pos0_y"]].iloc[test_idx].values

    # Uložení do výsledného DataFrame
    vysledky_df[f"skutecna_{prefix}_x"] = y_true[f"target_{prefix}_x"]
    vysledky_df[f"skutecna_{prefix}_y"] = y_true[f"target_{prefix}_y"]
    vysledky_df[f"nejvetsi_vaha_{prefix}_x"] = y_pred_nn[:, 0]
    vysledky_df[f"nejvetsi_vaha_{prefix}_y"] = y_pred_nn[:, 1]
    vysledky_df[f"predikce_{prefix}_x"] = y_pred_rf[:, 0]
    vysledky_df[f"predikce_{prefix}_y"] = y_pred_rf[:, 1]

# Uložení do CSV
output_path = "data/predikce_bodu_optuna.csv"
import os
os.makedirs(os.path.dirname(output_path), exist_ok=True)
vysledky_df.to_csv(output_path, index=False)
print(f"Predikce uloženy do {output_path}")