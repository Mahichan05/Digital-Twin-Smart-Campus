import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("campus_data.csv")

# Prepare data
X = data[['Students','Temperature']]
y = data['Electricity']

# Train model
model = LinearRegression()
model.fit(X,y)

# Page title
st.title("Smart Campus Digital Twin")
st.write("AI system to predict campus electricity consumption")

# Sliders
students = st.slider("Number of Students",700,1000,900)
temperature = st.slider("Temperature (°C)",25,40,32)

# Prediction
prediction = model.predict([[students,temperature]])

st.subheader("Predicted Electricity Usage")
st.success(str(round(prediction[0],2)) + " kWh")

# Sustainability score
score = 100 - (prediction[0] / 20)

st.subheader("Sustainability Score")

if score > 70:
    st.success("Good Sustainability 🌱")
elif score > 50:
    st.warning("Moderate Sustainability ⚠️")
else:
    st.error("High Energy Consumption ❌")

st.write("Score:", round(score,2), "/ 100")

# Energy saving suggestion
st.subheader("Energy Recommendation")

if prediction[0] > 1250:
    st.write("⚡ High electricity usage predicted.")
    st.write("Suggestion: Reduce AC usage and optimize lighting.")
else:
    st.write("⚡ Energy usage within normal range.")

# Electricity trend graph
st.subheader("Electricity Usage Trend")

plt.plot(data['Electricity'])
plt.xlabel("Days")
plt.ylabel("Electricity (kWh)")
plt.title("Campus Electricity Trend")

st.pyplot(plt)

# Show dataset
st.subheader("Dataset Preview")
st.dataframe(data)