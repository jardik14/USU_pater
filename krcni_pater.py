import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import optuna

df = pd.read_csv("data-recovery.csv")

# Uložíme regresory a názvy featur
regressors = {}
feature_names_per_kp = {}

for kp in range(23):
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

    # Uložení názvů featur pro pozdější použití
    feature_names_per_kp[kp] = features

    X = df[features]
    y = df[[f"target_kp{kp}_x", f"target_kp{kp}_y"]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    base_model = RandomForestRegressor(n_estimators=100, random_state=42)
    multi_output_model = MultiOutputRegressor(base_model)
    multi_output_model.fit(X_train, y_train)

    regressors[f"kp{kp}"] = multi_output_model

    y_pred = multi_output_model.predict(X_test)
    mse_x = mean_squared_error(y_test.iloc[:, 0], y_pred[:, 0])
    mse_y = mean_squared_error(y_test.iloc[:, 1], y_pred[:, 1])
    r2_x = r2_score(y_test.iloc[:, 0], y_pred[:, 0])
    r2_y = r2_score(y_test.iloc[:, 1], y_pred[:, 1])

    print(f"=== kp{kp} ===")
    print(f"X - MSE: {mse_x:.2f}, R2: {r2_x:.2f}")
    print(f"Y - MSE: {mse_y:.2f}, R2: {r2_y:.2f}")
    print("--------")


# Řádek k vizualizaci (např. první)
row = df.iloc[0]

# Inicializace prázdných seznamů
target_points = []
neural_points = []
rf_points = []

for kp in range(23):
    # Ground truth
    tx = row[f"target_kp{kp}_x"]
    ty = row[f"target_kp{kp}_y"]
    target_points.append((tx, ty))

    # Neuronka (predikce 0)
    nx = row[f"pred_kp{kp}_pos0_x"]
    ny = row[f"pred_kp{kp}_pos0_y"]
    neural_points.append((nx, ny))

    # Regrese
    model = regressors[f"kp{kp}"]
    feature_names = feature_names_per_kp[kp]
    feature_values = [row[f] for f in feature_names]
    X_input = pd.DataFrame([feature_values], columns=feature_names)
    x_pred, y_pred = model.predict(X_input)[0]
    rf_points.append((x_pred, y_pred))

# Převod na numpy pro pohodlnější práci
target_points = np.array(target_points)
neural_points = np.array(neural_points)
rf_points = np.array(rf_points)

# --- Vykreslení ---
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
titles = ["Ground Truth", "Neuronka (pred 0)", "Regrese (Random Forest)"]
data_sets = [target_points, neural_points, rf_points]
colors = ["green", "blue", "red"]

for ax, title, data, color in zip(axs, titles, data_sets, colors):
    ax.scatter(data[:, 0], data[:, 1], c=color, label=title)
    for i, (x, y) in enumerate(data):
        ax.text(x + 1, y + 1, str(i), fontsize=8, color=color)  # číslo bodu
    ax.set_title(title)
    ax.set_xlim(0, df[[f"target_kp{kp}_x" for kp in range(23)]].max().max() + 20)
    ax.set_ylim(0, df[[f"target_kp{kp}_y" for kp in range(23)]].max().max() + 20)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.grid(True)

plt.tight_layout()
plt.show()
