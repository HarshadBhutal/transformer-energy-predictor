import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


cont_df = pd.read_csv("Energy Prediction Forecasting Dataset/continuous dataset.csv")
train_df = pd.read_excel("Energy Prediction Forecasting Dataset/train_dataframes.xlsx")
test_df = pd.read_excel("Energy Prediction Forecasting Dataset/test_dataframes.xlsx")
weekly_df = pd.read_csv("Energy Prediction Forecasting Dataset/weekly pre-dispatch forecast.csv")

datasets = {
    "continuous_df": cont_df,
    "train_df": train_df,
    "test_df": test_df,
    "weekly_df": weekly_df
}

for name, df in datasets.items():
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        df.sort_values('datetime', inplace=True)
        df.reset_index(drop=True, inplace=True)

for name, df in datasets.items():
    print(f"\nMissing values in {name}:")
    print(df.isnull().sum())

print("\nShapes of datasets:")
for name, df in datasets.items():
    print(name, ":", df.shape)

print("\nPreview continuous dataset:")
print(cont_df.head())


df = train_df.copy()

target = "DEMAND"

features = df.select_dtypes(include=[np.number]).columns.tolist()

features.remove(target)

X = df[features]
y = df[target]


X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_val)

mae = mean_absolute_error(y_val, y_pred)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))

print("Model Performance:")
print("MAE:", mae)
print("RMSE:", rmse)

comparison = pd.DataFrame({
    "Actual": y_val.values,
    "Predicted": y_pred
})
print("\nComparison:")
print(comparison.head())




