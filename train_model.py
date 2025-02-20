import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Load the dataset
data = pd.read_csv('diabetes.csv')

# Splitting features and target variable
X = data.drop(columns=['Outcome'])
y = data['Outcome']

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Naïve Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Save the trained model
with open('diabetes_model.sav', 'wb') as file:
    pickle.dump(model, file)

print("Model saved successfully as 'diabetes_model.sav'!")
