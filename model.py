import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv("campus_data.csv")

X = data[['Students', 'Temperature']]
y = data['Electricity']

model = LinearRegression()
model.fit(X, y)

prediction = model.predict([[900, 32]])

print("Predicted Electricity Usage:", prediction)