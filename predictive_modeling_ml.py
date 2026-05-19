# predictive_modeling_ml.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Load Iris Dataset
iris = load_iris()

# Input and Output Data
X = iris.data
y = iris.target

# Split Dataset into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Random Forest Model
model = RandomForestClassifier()

# Train the Model
model.fit(X_train, y_train)

# Predict Test Data
y_pred = model.predict(X_test)

# Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# Create Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Display Confusion Matrix
sns.heatmap(cm, annot=True, cmap="Blues")

plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.title("Confusion Matrix")

# Save Output Image
plt.savefig("confusion_matrix_output.png")

# Show Graph
plt.show()

# Print Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))