import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['Species'] = iris.target
    return df, iris.target_names

df, target_name = load_data()

# --- Manual KNN Classifier ---
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def knn_predict(X_train, y_train, X_test, k=3):
    predictions = []
    for test_point in X_test:
        distances = []
        for i in range(len(X_train)):
            dist = euclidean_distance(test_point, X_train[i])
            distances.append((dist, y_train[i]))
        distances.sort(key=lambda x: x[0])
        neighbors = distances[:k]
        classes = [label for _, label in neighbors]
        predicted_class = max(set(classes), key=classes.count)
        predictions.append(predicted_class)
    return predictions

# --- Sidebar sliders for input ---
sepal_length = st.sidebar.slider(
    "Sepal length (cm)", 
    float(df['sepal length (cm)'].min()), 
    float(df['sepal length (cm)'].max())
)

sepal_width = st.sidebar.slider(
    "Sepal width (cm)", 
    float(df['sepal width (cm)'].min()), 
    float(df['sepal width (cm)'].max())
)

petal_length = st.sidebar.slider(
    "Petal length (cm)", 
    float(df['petal length (cm)'].min()), 
    float(df['petal length (cm)'].max())
)

petal_width = st.sidebar.slider(
    "Petal width (cm)", 
    float(df['petal width (cm)'].min()), 
    float(df['petal width (cm)'].max())
)

input_data = [[sepal_length, sepal_width, petal_length, petal_width]]

# --- Manual prediction using KNN ---
X_train = df.iloc[:, :-1].values
y_train = df['Species'].values

prediction = knn_predict(X_train, y_train, input_data, k=5)
predicted_species = target_name[prediction[0]]

# --- Output ---
st.title("Iris Species Predictor 🌸")
st.write(f"### The predicted species is: **{predicted_species}**")
