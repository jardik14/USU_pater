import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import optuna

# Načti dataset
df = pd.read_csv("data-recovery.csv")

regressors = {}
feature_names_per_kp = {}

# Optimalizace pro každý klíčový bod (kp)
for kp in range(23):
    print(f"Optimalizuji kp{kp}...")

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

    # Uložení názvů pro pozdější použití
    feature_names_per_kp[kp] = features

    # Data
    X = df[features]
    y = df[[f"target_kp{kp}_x", f"target_kp{kp}_y"]]

    # Rozdělení na trénovací/testovací sadu
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Definice optimalizační funkce
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
        mse = mean_squared_error(y_test, y_pred)
        return mse

    # Spuštění optimalizace
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30, show_progress_bar=True)

    print(f"  ↳ Nejlepší MSE: {study.best_value:.4f}")
    print(f"  ↳ Parametry: {study.best_params}")

    # Trénink finálního modelu s nejlepšími parametry
    best_rf = RandomForestRegressor(
        **study.best_params,
        random_state=42,
        n_jobs=-1
    )
    best_model = MultiOutputRegressor(best_rf)
    best_model.fit(X_train, y_train)

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
    print("🎉 Regrese je přesnější podle MSE!")
else:
    print("🧠 Neuronka (pos0) je přesnější podle MSE.")