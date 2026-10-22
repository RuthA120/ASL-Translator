import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. Load the dataset
df = pd.read_csv('asl_data.csv')

# Separate features (the 63 landmark coordinates) and labels (the letters)
X = df.drop('label', axis=1)
y = df['label']

# 2. Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train a Random Forest Classifier
print("Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Check accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model trained successfully! Accuracy: {acc * 100:.2f}%")

# 5. Save the trained model to a file
joblib.dump(model, 'asl_model.pkl')
print("Model saved as 'asl_model.pkl'.")