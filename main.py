# ================================================================
# LSTM Energy Forecasting (Continuous Dataset + Weekly Dispatch)
# Stable autoregressive forecasting included (no reshape errors)
# ================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# ---------------------------------------------------------------
# 1. LOAD DATASETS
# ---------------------------------------------------------------
continuous_path = "Energy Prediction Forecasting Dataset/continuous dataset.csv"
weekly_path     = "Energy Prediction Forecasting Dataset/weekly pre-dispatch forecast.csv"

cont_df = pd.read_csv(continuous_path)
weekly_df = pd.read_csv(weekly_path)

# ---------------------------------------------------------------
# 2. CLEAN + PREPARE DATETIME
# ---------------------------------------------------------------
cont_df["datetime"] = pd.to_datetime(cont_df["datetime"], errors="coerce")
cont_df = cont_df.sort_values("datetime").reset_index(drop=True)

weekly_df["datetime"] = pd.to_datetime(weekly_df["datetime"], errors="coerce")
weekly_df = weekly_df.sort_values("datetime").reset_index(drop=True)

# ---------------------------------------------------------------
# 3. FEATURE ENGINEERING
# ---------------------------------------------------------------
TARGET = "nat_demand"

cont_df["hour"] = cont_df["datetime"].dt.hour
cont_df["dayofweek"] = cont_df["datetime"].dt.dayofweek
cont_df["month"] = cont_df["datetime"].dt.month
cont_df["is_weekend"] = cont_df["dayofweek"].isin([5,6]).astype(int)

# Create lags
cont_df["lag_1"] = cont_df[TARGET].shift(1)
cont_df["lag_24"] = cont_df[TARGET].shift(24)
cont_df["lag_168"] = cont_df[TARGET].shift(168)

# Rolling means
cont_df["ma_24"] = cont_df[TARGET].rolling(24).mean().shift(1)
cont_df["ma_168"] = cont_df[TARGET].rolling(168).mean().shift(1)

# Remove NaN rows created by lags
cont_df = cont_df.dropna().reset_index(drop=True)

# ---------------------------------------------------------------
# 4. SELECT FEATURES
# ---------------------------------------------------------------
feature_cols = [
    "hour","dayofweek","month","is_weekend",
    "lag_1","lag_24","lag_168","ma_24","ma_168"
]

dataset = cont_df[feature_cols + [TARGET]]

# ---------------------------------------------------------------
# 5. SCALE FEATURES
# ---------------------------------------------------------------
scaler = MinMaxScaler()
scaled = scaler.fit_transform(dataset)

joblib.dump(scaler, "scaler.gz")

scaled_df = pd.DataFrame(scaled, columns=feature_cols + [TARGET])

# ---------------------------------------------------------------
# 6. CREATE LSTM SEQUENCES
# ---------------------------------------------------------------
SEQ_LEN = 24

def create_sequences(data, seq_len=24):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len, :-1])
        y.append(data[i+seq_len, -1])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_df.values, SEQ_LEN)

# ---------------------------------------------------------------
# 7. TRAIN / VAL SPLIT
# ---------------------------------------------------------------
split = int(0.80 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# ---------------------------------------------------------------
# 8. BUILD LSTM MODEL
# ---------------------------------------------------------------
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, len(feature_cols))),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dense(1)
])
model.compile(optimizer="adam", loss="mse")
model.summary()

# ---------------------------------------------------------------
# 9. TRAIN THE MODEL
# ---------------------------------------------------------------
es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[es],
    verbose=1
)

model.save("lstm_model.keras")

# ---------------------------------------------------------------
# 10. EVALUATE ON VALIDATION SET
# ---------------------------------------------------------------
val_pred_scaled = model.predict(X_val)

dummy_pred = np.zeros((len(val_pred_scaled), len(feature_cols)+1))
dummy_pred[:, -1] = val_pred_scaled[:,0]
val_pred = scaler.inverse_transform(dummy_pred)[:, -1]

dummy_true = np.zeros((len(y_val), len(feature_cols)+1))
dummy_true[:, -1] = y_val
val_true = scaler.inverse_transform(dummy_true)[:, -1]

mae = mean_absolute_error(val_true, val_pred)
rmse = np.sqrt(mean_squared_error(val_true, val_pred))

print("\nValidation MAE:", mae)
print("Validation RMSE:", rmse)

# ---------------------------------------------------------------
# 11. REAL FORECASTING USING WEEKLY DISPATCH
# ---------------------------------------------------------------
future_steps = len(weekly_df)
future_dates = weekly_df["datetime"].reset_index(drop=True)

# last sequence from training
scaled_values = scaled_df.values
last_seq = scaled_values[-SEQ_LEN:]
last_features = last_seq[:, :-1]  # drop target column

current_window = last_features.copy()
predictions_scaled = []

for i in range(future_steps):
    # LSTM input
    x_input = current_window.reshape(1, SEQ_LEN, len(feature_cols))
    pred_scaled = model.predict(x_input, verbose=0)[0,0]
    predictions_scaled.append(pred_scaled)

    # Construct next feature row
    new_row = current_window[-1].copy()

    # update lag features
    new_row[feature_cols.index("lag_1")] = pred_scaled
    new_row[feature_cols.index("lag_24")] = current_window[-1, feature_cols.index("lag_1")]
    new_row[feature_cols.index("lag_168")] = current_window[-1, feature_cols.index("lag_24")]

    # roll window
    current_window = np.vstack([current_window[1:], new_row])

# Inverse transform forecasts
pred_scaled_arr = np.array(predictions_scaled).reshape(-1,1)
dummy = np.zeros((len(pred_scaled_arr), len(feature_cols)+1))
dummy[:,-1] = pred_scaled_arr[:,0]
final_predictions = scaler.inverse_transform(dummy)[:, -1]

forecast_df = pd.DataFrame({
    "datetime": future_dates,
    "predicted_nat_demand": final_predictions
})

print("\nForecast Preview:")
print(forecast_df.head())

forecast_df.to_csv("weekly_forecast_output.csv", index=False)
print("\nSaved forecast to weekly_forecast_output.csv")

# ---------------------------------------------------------------
# 12. PLOT FINAL FORECAST
# ---------------------------------------------------------------
plt.figure(figsize=(12,5))
plt.plot(forecast_df["datetime"], forecast_df["predicted_nat_demand"], label="Predicted Weekly Forecast", linewidth=2)
plt.title("Weekly Forecast of nat_demand using LSTM")
plt.xlabel("Datetime")
plt.ylabel("nat_demand")
plt.legend()
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

