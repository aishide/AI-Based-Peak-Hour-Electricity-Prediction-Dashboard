import pandas as pd
import numpy as np

dorms = ["Dorm_A", "Dorm_B", "Dorm_C"] 
hours = pd.date_range(start="2025-01-01", periods=24*14, freq="h")

data = []
for dorm in dorms:
    for time in hours:
        hour = time.hour

        if 18 <= hour <= 22:
            usage = np.random.uniform(30, 45)
        elif 6 <= hour <= 9:
            usage = np.random.uniform(20, 30)
        else:
            usage = np.random.uniform(10, 20)

        data.append([time, dorm, round(usage, 2)])

df = pd.DataFrame(data, columns=["timestamp", "dorm", "consumption_kwh"])
df.to_csv("./data/electricity_data.csv", index=False)

print("Data for multiple dorms generated successfully!")
