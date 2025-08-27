import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Training data
X_train = np.array([
    [1, 2],  # Class A
    [2, 1],  # Class A
    [3, 2],  # Class A
    [6, 6],  # Class B
    [7, 7],  # Class B
    [8, 6],  # Class B
])
y_train = ['A', 'A', 'A', 'B', 'B', 'B']

# Test point and actual label
X_test = np.array([[5, 5]])
y_test = ['B']

# Train KNN model (k=3)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predict
prediction = knn.predict(X_test)[0]
print(f"Predicted class for {X_test[0]}: {prediction}, Actual: {y_test[0]}")

# Accuracy (just for this single test point)
accuracy = 1.0 if prediction == y_test[0] else 0.0

# Visualization
plt.figure(figsize=(6, 6))

# Plot training data
for i, label in enumerate(y_train):
    if label == 'A':
        plt.scatter(X_train[i, 0], X_train[i, 1], color='blue', label='Train A' if i == 0 else "")
    else:
        plt.scatter(X_train[i, 0], X_train[i, 1], color='green', label='Train B' if i == 3 else "")

# Plot test point
plt.scatter(X_test[0, 0], X_test[0, 1], color='red', marker='x', s=100, label='Test Point')

# Draw lines to nearest neighbors
neighbors = knn.kneighbors(X_test, return_distance=False)[0]
for n in neighbors:
    plt.plot([X_test[0, 0], X_train[n, 0]], [X_test[0, 1], X_train[n, 1]], 'k--')

plt.title(f"KNN (K=3): Predicted = {prediction}, Accuracy = {accuracy*100:.1f}%")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()

