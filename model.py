import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("data/electricity_data.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"])

df["hour"] = df["timestamp"].dt.hour

df["smoothed_consumption"] = df["consumption_kwh"].rolling(window=3).mean()
df = df.dropna()

X = df[["hour"]]   
y = df["smoothed_consumption"]  

model = LinearRegression()
model.fit(X, y)

evening_hours = pd.DataFrame({"hour": [18, 19, 20, 21, 22]})
predicted_peak = model.predict(evening_hours)

print("🔮 Predicted Evening Peak Consumption (kWh):")
for h, val in zip(evening_hours["hour"], predicted_peak):
    print(f"Hour {h}: {val:.2f} kWh")
