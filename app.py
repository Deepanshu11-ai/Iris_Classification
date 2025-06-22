import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

st.set_page_config(page_title="🌸 Iris Species Predictor", layout="centered")

# Load data with caching
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['Species'] = iris.target
    return df, iris.target_names

df, target_name = load_data()

# Manual KNN
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def knn_predict(X_train, y_train, X_test, k=5):
    predictions = []
    confidences = []
    for test_point in X_test:
        distances = []
        for i in range(len(X_train)):
            dist = euclidean_distance(test_point, X_train[i])
            distances.append((dist, y_train[i]))
        distances.sort(key=lambda x: x[0])
        neighbors = distances[:k]
        classes = [label for _, label in neighbors]
        predicted_class = max(set(classes), key=classes.count)
        confidence = classes.count(predicted_class) / k
        predictions.append(predicted_class)
        confidences.append(confidence)
    return predictions, confidences

# UI Elements
st.title("🌼 Iris Species Classifier")
st.markdown("Predict the **species** of an Iris flower using a hand-built KNN algorithm! 🔍")

with st.expander("🔧 Input Flower Features"):
    sepal_length = st.slider("Sepal length (cm)", float(df['sepal length (cm)'].min()), float(df['sepal length (cm)'].max()))
    sepal_width = st.slider("Sepal width (cm)", float(df['sepal width (cm)'].min()), float(df['sepal width (cm)'].max()))
    petal_length = st.slider("Petal length (cm)", float(df['petal length (cm)'].min()), float(df['petal length (cm)'].max()))
    petal_width = st.slider("Petal width (cm)", float(df['petal width (cm)'].min()), float(df['petal width (cm)'].max()))

input_data = [[sepal_length, sepal_width, petal_length, petal_width]]

# Predict
X_train = df.iloc[:, :-1].values
y_train = df['Species'].values

prediction, confidence = knn_predict(X_train, y_train, input_data, k=5)
predicted_species = target_name[prediction[0]]

# Output
st.subheader("📢 Prediction Result:")
st.success(f"**Predicted species:** {predicted_species} 🌸")
st.write(f"Confidence: `{confidence[0]*100:.1f}%`")

# Plot: Input vs Dataset
st.subheader("🌈 Feature Comparison (Petal Length vs Width)")
fig, ax = plt.subplots()
colors = ['red', 'green', 'blue']
for i in range(3):
    species_data = df[df['Species'] == i]
    ax.scatter(species_data['petal length (cm)'], species_data['petal width (cm)'], label=target_name[i], alpha=0.6, c=colors[i])
ax.scatter(petal_length, petal_width, color='black', s=100, label="Your Input", marker='X')
ax.set_xlabel("Petal Length (cm)")
ax.set_ylabel("Petal Width (cm)")
ax.legend()
st.pyplot(fig)

# Confidence Pie Chart
st.subheader("🧠 Prediction Confidence Breakdown")
confidence_labels = [target_name[i] for i in range(3)]
confidence_values = [0] * 3
for i in range(3):
    confidence_values[i] = sum(
        1 for j in range(len(X_train))
        if euclidean_distance(input_data[0], X_train[j]) < 0.5 and y_train[j] == i
    )
# Normalize the data
if sum(confidence_values) > 0:
    confidence_values = [v / sum(confidence_values) for v in confidence_values]
else:
    confidence_values = [1 if i == prediction[0] else 0 for i in range(3)]

fig2, ax2 = plt.subplots()
ax2.pie(confidence_values, labels=confidence_labels, autopct="%1.1f%%", startangle=90, colors=colors)
ax2.axis('equal')
st.pyplot(fig2)

# Footer part 
st.markdown("---")
st.caption("Made with 💖 by Deepanshu — manual ML > blackbox 😎")
