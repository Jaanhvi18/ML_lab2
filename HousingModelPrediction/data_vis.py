import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_excel("predictions.xlsx")

# Model 1
plt.figure(figsize=(10, 6))
plt.scatter(
    data["transaction_price_2020"],
    data["Predicted_Price_Model1"],
    label="Model 1",
    alpha=0.6,
)
plt.xlabel("Actual Transaction Price")
plt.ylabel("Predicted Price (Model 1)")
plt.title("Actual vs. Predicted Prices (Model 1)")
plt.legend()
plt.grid(True)
plt.show()

# Model 1
plt.figure(figsize=(10, 6))
plt.scatter(
    data["transaction_price_2020"],
    data["Predicted_Price_Model2"],
    label="Model 2",
    alpha=0.6,
)
plt.xlabel("Actual Transaction Price")
plt.ylabel("Predicted Price (Model 2)")
plt.title("Actual vs. Predicted Prices (Model 2)")
plt.legend()
plt.grid(True)
plt.show()
