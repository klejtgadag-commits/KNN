# Iris Flower Classification using K-Nearest Neighbors (KNN)

A Machine Learning project demonstrating the application of the K-Nearest Neighbors (KNN) algorithm on the classic Iris dataset using Python and `scikit-learn`.

## 📌 Project Overview

This project implements a KNN classifier to predict flower species based on sepal and petal dimensions. It covers data loading, feature extraction, dataset split, model fitting, inference, and evaluation.

### Algorithm Highlights
- **Lazy Learning**: Memorizes the training dataset rather than explicitly training parameters.
- **Distance Metric**: Uses Minkowski distance ($p=2$, standard Euclidean distance) to compute proximity between test data points and training data points.
- **Classification Strategy**: Assigns class labels using a majority vote among the $K$ nearest neighbors.
- **Parameter Selection**: $K=3$ (odd integer selected to break potential tie-breaks).

---

## 🛠️ Tech Stack & Dependencies

* **Language**: Python 3.x
* **Data Processing & Analysis**: `pandas`
* **Machine Learning**: `scikit-learn`
* **Data Visualization**: `matplotlib`

---

## 📊 Dataset Information

* **Source**: `sklearn.datasets.load_iris`
* **Instances**: 150 (50 instances per class)
* **Classes**: 
  1. `Setosa` (0)
  2. `Versicolor` (1)
  3. `Virginica` (2)
* **Features**:
  * `sepal length (cm)`
  * `sepal width (cm)`
  * `petal length (cm)`
  * `petal width (cm)`

---

## ⚙️ Model Pipeline & Execution

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. Load Dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# 2. Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Initialize Model
knn = KNeighborsClassifier(n_neighbors=3)

# 4. Train Model
knn.fit(X_train, y_train)

# 5. Predict & Evaluate
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%")
