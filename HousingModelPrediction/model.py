import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
import csv
from sklearn.preprocessing import StandardScaler


data = pd.read_excel("transaction_prices_2000.xlsx")

X = data[["squarefeet_floorspace", "bedrooms", "rooms"]]

y = data["transaction_price_2020"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


degree = 2
alpha = 1.0


poly_features = PolynomialFeatures(degree=degree)
X_poly = poly_features.fit_transform(X)

model = Ridge(alpha=alpha)
model.fit(X_poly, y)
X_all_poly = poly_features.transform(X)
predictions = model.predict(X_all_poly)
data["Hedonic Model 1 Prediction"] = predictions
error_model1 = y - predictions
data["Error_Model1"] = error_model1
degree = 3
alpha = 0.1

poly_features = PolynomialFeatures(degree=degree)
X_poly = poly_features.fit_transform(X)

model = Ridge(alpha=alpha)
model.fit(X_poly, y)


X_all_poly = poly_features.transform(X)
predictions = model.predict(X_all_poly)

data["Hedonic Model 2 Prediction"] = predictions
error_model2 = y - predictions
data["Error_Model2"] = error_model2
data.to_excel("/Users/jaanhviagarwal/Desktop/predictions_with_errors.xlsx", index=False)
