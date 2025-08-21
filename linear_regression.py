import pandas as pd
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

# Load the CSV
data = pd.read_csv('homeprices.csv')
print(data)

# Separate features and labels
y = data[['price']]
x = data[['area']]

# Train the model
model = LinearRegression()
model.fit(x, y)

# Predict and calculate accuracy
y_train_pred = model.predict(x)
r2 = r2_score(y, y_train_pred)
print("Accuracy (R² score):", r2)
