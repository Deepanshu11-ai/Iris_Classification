# Iris_Classification
# 🌸 Iris Species Predictor using Manual KNN + Streamlit

This is a fun little machine learning project where we predict the species of an Iris flower based on its measurements using a **manually implemented K-Nearest Neighbors (KNN)** algorithm — no scikit-learn classifiers involved! The app is built using **Streamlit** for a clean and interactive UI.

---

## 🚀 Live Demo

> You can run it locally using Streamlit. Scroll down for setup instructions!

---

## 📌 Features

- 🔍 Predict Iris species (Setosa, Versicolor, Virginica)
- 🎛 Interactive sliders for flower features
- 🤖 Manual implementation of KNN algorithm
- 🧼 Clean and minimal UI with Streamlit

---

## 🧠 How It Works

1. The app loads the famous **Iris dataset**.
2. A **custom KNN algorithm** calculates the Euclidean distance between the user input and the dataset.
3. It picks the **K nearest neighbors** (default `k=5`) and uses majority voting to classify.
4. Streamlit lets you input Sepal & Petal dimensions and see the prediction live.


## 🛠 Tech Stack

- Python 🐍
- Streamlit 🌐
- NumPy 📊
- Pandas 🐼

---

## 📦 Installation & Running

1. **Clone the repo**
```bash
git clone https://github.com/your-username/iris-knn-streamlit.git
cd iris-knn-streamlit
