import pandas as pd
from sklearn.preprocessing import LabelEncoder

city = pd.DataFrame([[1000, "Roma"], [2000, "Torino"], [3000, "Milano"], [4000, "Roma"], [5000, "Torino"], [6000, "Milano"], [5, "Carignano"]], columns=["Abitanti", "Citta"])
city_2 = pd.DataFrame([[1000, "Roma"]], columns=["Abitanti", "Citta"])

label_encoder = LabelEncoder()

print(city)
label_encoder.fit(city["Citta"])
city_encoded = label_encoder.transform(city_2["Citta"])
print(city_encoded)