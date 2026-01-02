import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 1. Load the Dataset
dataset = pd.read_csv('data.csv')
X = dataset.iloc[:, :-1].values # Years of Experience
y = dataset.iloc[:, 1].values   # Salary

# 2. Split into Training set and Test set
# We use 80% of data to train, and 20% to test how accurate we are
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 3. Train the Model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

print("Training Complete. Model is ready.")

# 4. Make a Prediction
# Let's predict the salary for someone with 12 years of experience
new_prediction = regressor.predict([[12]])
print(f"Predicted Salary for 12 Years Experience: ${new_prediction[0]:.2f}")

# 5. Visualize the Results
plt.scatter(X_train, y_train, color='red', label='Real Data')
plt.plot(X_train, regressor.predict(X_train), color='blue', label='Prediction Line')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.savefig('prediction_graph.png') # Saves the graph as an image
print("Graph saved as 'prediction_graph.png'")
