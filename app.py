import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Predictive Analytics Dashboard", layout="wide")

st.title("Predictive Analytics Using Historical Data")

df = pd.read_csv("airline-passengers.csv")

df['Month'] = pd.to_datetime(df['Month'])

df['Time'] = range(len(df))

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Dataset Information")
st.write(df.describe())

sns.set()

st.subheader("Passenger Trend Over Time")

fig1, ax1 = plt.subplots(figsize=(12,6))
ax1.plot(df['Month'], df['Passengers'])
ax1.set_title("Airline Passengers Over Time")
ax1.set_xlabel("Year")
ax1.set_ylabel("Passengers")

st.pyplot(fig1)

st.subheader("Passenger Distribution")

fig2, ax2 = plt.subplots(figsize=(8,5))
sns.histplot(df['Passengers'], bins=20, ax=ax2)
ax2.set_title("Passenger Distribution")

st.pyplot(fig2)

st.subheader("Boxplot Analysis")

fig3, ax3 = plt.subplots(figsize=(8,5))
sns.boxplot(y=df['Passengers'], ax=ax3)
ax3.set_title("Boxplot of Passengers")

st.pyplot(fig3)

df['Year'] = df['Month'].dt.year

yearly_avg = df.groupby('Year')['Passengers'].mean()

st.subheader("Yearly Average Passengers")

fig4, ax4 = plt.subplots(figsize=(12,6))
yearly_avg.plot(kind='bar', ax=ax4)
ax4.set_title("Average Passengers Per Year")
ax4.set_xlabel("Year")
ax4.set_ylabel("Average Passengers")

st.pyplot(fig4)

X = df[['Time']]
y = df['Passengers']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

results = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': predictions
})

st.subheader("Prediction Results")
st.dataframe(results.head())

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

st.subheader("Model Accuracy")

st.write(f"Mean Absolute Error: {mae:.2f}")
st.write(f"Mean Squared Error: {mse:.2f}")
st.write(f"R2 Score: {r2:.2f}")

st.subheader("Actual vs Predicted Passengers")

fig5, ax5 = plt.subplots(figsize=(12,6))

ax5.plot(y_test.values, label='Actual Passengers')
ax5.plot(predictions, label='Predicted Passengers')

ax5.set_title("Actual vs Predicted Passengers")
ax5.set_xlabel("Time")
ax5.set_ylabel("Passengers")

ax5.legend()

st.pyplot(fig5)

future_time = [[150]]

future_prediction = model.predict(future_time)

st.subheader("Future Prediction")

st.success(f"Predicted Future Passenger Count: {future_prediction[0]:.2f}")