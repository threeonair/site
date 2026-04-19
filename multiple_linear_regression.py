import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────
# 1. Load Dataset Online (No Download)
# ─────────────────────────────────────────────
print("Fetching Metro Interstate Traffic Volume dataset from UCI...")
dataset = fetch_ucirepo(id=492)

X_raw = dataset.data.features
y     = dataset.data.targets.squeeze()

print(f"Dataset loaded: {X_raw.shape[0]} rows, {X_raw.shape[1]} features")
print(f"\nFeatures: {list(X_raw.columns)}")
print(f"Target  : traffic_volume\n")

# ─────────────────────────────────────────────
# 2. Feature Engineering
# ─────────────────────────────────────────────
df = X_raw.copy()

# Parse datetime features
df["date_time"]   = pd.to_datetime(df["date_time"])
df["hour"]        = df["date_time"].dt.hour
df["day_of_week"] = df["date_time"].dt.dayofweek
df["month"]       = df["date_time"].dt.month
df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)

# Encode categorical: holiday (is it a holiday or not)
df["is_holiday"] = (df["holiday"] != "None").astype(int)

# Select numeric features for regression
feature_cols = ["temp", "rain_1h", "snow_1h", "clouds_all",
                "hour", "day_of_week", "month", "is_weekend", "is_holiday"]

X = df[feature_cols]
print("=" * 55)
print("Feature Summary")
print("=" * 55)
print(X.describe().round(2))

# ─────────────────────────────────────────────
# 3. Train-Test Split
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining samples : {X_train.shape[0]}")
print(f"Testing  samples : {X_test.shape[0]}")

# ─────────────────────────────────────────────
# 4. Feature Scaling
# ─────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# 5. Train Model
# ─────────────────────────────────────────────
model = LinearRegression()
model.fit(X_train_scaled, y_train)

print("\n" + "=" * 55)
print("Model Coefficients")
print("=" * 55)
for name, coef in zip(feature_cols, model.coef_):
    print(f"  {name:<20}: {coef:>10.4f}")
print(f"  {'Intercept':<20}: {model.intercept_:>10.4f}")

# ─────────────────────────────────────────────
# 6. Predictions & Evaluation
# ─────────────────────────────────────────────
y_pred = model.predict(X_test_scaled)

mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print("\n" + "=" * 55)
print("Evaluation Metrics")
print("=" * 55)
print(f"  MAE  (Mean Absolute Error)     : {mae:.2f}")
print(f"  MSE  (Mean Squared Error)      : {mse:.2f}")
print(f"  RMSE (Root Mean Squared Error) : {rmse:.2f}")
print(f"  R²   (Coefficient of Det.)     : {r2:.4f}")

# ─────────────────────────────────────────────
# 7. Visualization Plots
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Multiple Linear Regression\nMetro Interstate Traffic Volume",
             fontsize=15, fontweight="bold")

# --- Plot 1: Actual vs Predicted ---
ax1 = axes[0, 0]
ax1.scatter(y_test, y_pred, alpha=0.3, color="steelblue", s=10)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
ax1.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Fit")
ax1.set_xlabel("Actual Traffic Volume")
ax1.set_ylabel("Predicted Traffic Volume")
ax1.set_title("Actual vs Predicted")
ax1.legend()

# --- Plot 2: Residuals Distribution ---
ax2 = axes[0, 1]
residuals = y_test - y_pred
ax2.hist(residuals, bins=60, color="coral", edgecolor="white", alpha=0.8)
ax2.axvline(0, color="red", linestyle="--", linewidth=2)
ax2.set_xlabel("Residuals")
ax2.set_ylabel("Frequency")
ax2.set_title("Residuals Distribution")

# --- Plot 3: Residuals vs Predicted ---
ax3 = axes[1, 0]
ax3.scatter(y_pred, residuals, alpha=0.2, color="mediumseagreen", s=10)
ax3.axhline(0, color="red", linestyle="--", linewidth=2)
ax3.set_xlabel("Predicted Traffic Volume")
ax3.set_ylabel("Residuals")
ax3.set_title("Residuals vs Predicted")

# --- Plot 4: Feature Coefficients ---
ax4 = axes[1, 1]
colors = ["steelblue" if c > 0 else "coral" for c in model.coef_]
bars = ax4.barh(feature_cols, model.coef_, color=colors, edgecolor="white")
ax4.axvline(0, color="black", linewidth=0.8)
ax4.set_xlabel("Coefficient Value")
ax4.set_title("Feature Coefficients")
for bar, coef in zip(bars, model.coef_):
    ax4.text(
        coef + (5 if coef >= 0 else -5),
        bar.get_y() + bar.get_height() / 2,
        f"{coef:.1f}",
        va="center",
        ha="left" if coef >= 0 else "right",
        fontsize=8
    )

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/regression_plots.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlots saved to regression_plots.png")
