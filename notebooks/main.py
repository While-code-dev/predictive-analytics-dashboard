import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("data/airline-passengers.csv")

df['Month'] = pd.to_datetime(df['Month'])

df['Time'] = range(len(df))

print("FIRST 5 ROWS")
print(df.head())

print("\nDATA INFO")
print(df.info())

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nSTATISTICAL SUMMARY")
print(df.describe())

sns.set()

plt.figure(figsize=(12,6))
plt.plot(df['Month'], df['Passengers'])
plt.title("Airline Passengers Over Time")
plt.xlabel("Year")
plt.ylabel("Passengers")
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(df['Passengers'], bins=20)
plt.title("Passenger Distribution")
plt.xlabel("Passengers")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(y=df['Passengers'])
plt.title("Boxplot of Passengers")
plt.show()

df['Year'] = df['Month'].dt.year

yearly_avg = df.groupby('Year')['Passengers'].mean()

print("\nYEARLY AVERAGE PASSENGERS")
print(yearly_avg)

plt.figure(figsize=(12,6))
yearly_avg.plot(kind='bar')
plt.title("Average Passengers Per Year")
plt.xlabel("Year")
plt.ylabel("Average Passengers")
plt.show()

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

print("\nMODEL TRAINED SUCCESSFULLY")

predictions = model.predict(X_test)

results = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': predictions
})

print("\nPREDICTIONS")
print(results.head())

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nMODEL ACCURACY")

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

plt.figure(figsize=(12,6))
plt.plot(y_test.values, label='Actual Passengers')
plt.plot(predictions, label='Predicted Passengers')
plt.title("Actual vs Predicted Passengers")
plt.xlabel("Time")
plt.ylabel("Passengers")
plt.legend()
plt.show()

future_time = [[150]]

future_prediction = model.predict(future_time)

print("\nFUTURE PREDICTION")
print("Predicted Passenger Count:", future_prediction[0])