import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets

# Title and description
st.title("Machine Learning Algorithm Education")
st.markdown(
    """
    This page provides detailed information about various machine learning algorithms used in this application. 
    Select an algorithm to learn more about its characteristics, advantages, disadvantages, common use cases, 
    and an example implementation.
    """
)

# Algorithm selection
algorithm = st.selectbox(
    "Select an Algorithm to learn more:",
    [
        "Gaussian Naive Bayes (GaussianNB)",
        "Logistic Regression",
        "Random Forest",
        "SVM (RBF Kernel)",
        "k-Nearest Neighbors",
        "Decision Tree",
    ]
)

# Dictionary-based structure for details
algorithm_details = {
    "Gaussian Naive Bayes (GaussianNB)": {
        "Description": "A probabilistic classifier based on Bayes' theorem with strong independence assumptions. Assumes features follow a Gaussian (normal) distribution.",
        "Formula": r"P(y|X) = \frac{P(X|y) P(y)}{P(X)}",
        "Advantages": [
            "✅ Simple and fast",
            "✅ Works well with small datasets",
            "✅ Good for high-dimensional data",
            "✅ Performs well when features are normally distributed",
        ],
        "Disadvantages": [
            "⚠️ Assumes feature independence (often unrealistic)",
            "⚠️ Limited by Gaussian distribution assumption",
            "⚠️ May underperform when features are highly correlated",
        ],
        "Common Use Cases": [
            "🎯 Text classification",
            "🎯 Spam detection",
            "🎯 Medical diagnosis",
            "🎯 Real-time prediction scenarios",
        ],
        "Example Code": """
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample dataset
X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = GaussianNB()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
        """,
    },
    "Logistic Regression": {
        "Description": "A linear model for binary classification that predicts probabilities using the logistic function.",
        "Formula": r"f(x) = \frac{1}{1 + e^{-(w \cdot x + b)}}",
        "Advantages": [
            "✅ Simple and interpretable",
            "✅ Works well for linearly separable data",
            "✅ Efficient for large datasets",
        ],
        "Disadvantages": [
            "⚠️ Assumes linear relationship between features and target",
            "⚠️ Sensitive to multicollinearity",
        ],
        "Common Use Cases": [
            "🎯 Binary classification",
            "🎯 Medical diagnosis",
            "🎯 Customer churn prediction",
        ],
        "Example Code": """
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics: accuracy_score

# Sample dataset
X, y = datasets.load_iris(return_X_y=True)
X = X[y != 2]  # Binary classification
y = y[y != 2]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
        """,
    },
    "Random Forest": {
        "Description": "An ensemble learning method that builds multiple decision trees and combines their outputs.",
        "Formula": r"F(x) = \frac{1}{T} \sum_{t=1}^{T} h_t(x)",
        "Advantages": [
            "✅ Handles non-linear relationships",
            "✅ Resistant to overfitting (with enough trees)",
            "✅ Handles high-dimensional data",
        ],
        "Disadvantages": [
            "⚠️ Computationally expensive",
            "⚠️ Less interpretable than single decision trees",
        ],
        "Common Use Cases": [
            "🎯 Classification and regression tasks",
            "🎯 Feature importance analysis",
            "🎯 Fraud detection",
        ],
        "Example Code": """
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics: accuracy_score

# Sample dataset
X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
        """,
    },
    "SVM (RBF Kernel)": {
        "Description": "A powerful classifier that uses hyperplanes to separate data points in a high-dimensional space. The RBF kernel maps data points non-linearly to better handle complex relationships.",
        "Formula": r"K(x_i, x_j) = e^{-\gamma ||x_i - x_j||^2}",
        "Advantages": [
            "✅ Effective for high-dimensional data",
            "✅ Works well for non-linear boundaries",
        ],
        "Disadvantages": [
            "⚠️ Computationally intensive for large datasets",
            "⚠️ Requires careful parameter tuning (C and gamma)",
        ],
        "Common Use Cases": [
            "🎯 Image recognition",
            "🎯 Text categorization",
            "🎯 Bioinformatics",
        ],
        "Example Code": """
from sklearn.svm import SVC
from sklearn.model_selection: train_test_split
from sklearn.metrics: accuracy_score

# Sample dataset
X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = SVC(kernel='rbf', C=1, gamma=0.1)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
        """,
    },
    "k-Nearest Neighbors": {
        "Description": "A simple instance-based learning algorithm that predicts the label of a new data point based on the labels of its nearest neighbors.",
        "Formula": r"d(x, x') = \sqrt{\sum_{i=1}^{n} (x_i - x'_i)^2}",
        "Advantages": [
            "✅ Simple and intuitive",
            "✅ No training phase (lazy learning)",
        ],
        "Disadvantages": [
            "⚠️ Computationally expensive during prediction",
            "⚠️ Sensitive to noise and irrelevant features",
        ],
        "Common Use Cases": [
            "🎯 Pattern recognition",
            "🎯 Recommendation systems",
            "🎯 Medical diagnosis",
        ],
        "Example Code": """
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection: train_test_split
from sklearn.metrics: accuracy_score

# Sample dataset
X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
        """,
    },
    "Decision Tree": {
        "Description": "A tree-structured model that splits the data based on feature thresholds to make predictions.",
        "Formula": r"H(X) = -\sum_{i=1}^{n} p_i \log_2(p_i)",
        "Advantages": [
            "✅ Easy to interpret",
            "✅ Handles both categorical and numerical data",
            "✅ Requires little data preprocessing",
        ],
        "Disadvantages": [
            "⚠️ Prone to overfitting (without pruning)",
            "⚠️ May create biased splits with imbalanced data",
        ],
        "Common Use Cases": [
            "🎯 Classification and regression tasks",
            "🎯 Feature selection",
            "🎯 Risk analysis",
        ],
        "Example Code": """
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection: train_test_split
from sklearn.metrics: accuracy_score

# Sample dataset
X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
        """,
    },
}

# Retrieve details for the selected algorithm
details = algorithm_details[algorithm]

# Display sections with subheaders
st.subheader("Description:")
st.markdown(details["Description"])

advantages, disadvantages, common_use_cases = st.columns(3)

with advantages:
    st.subheader("Advantages:")
    st.markdown("\n".join(f"- {adv}" for adv in details["Advantages"]))
with disadvantages:
    st.subheader("Disadvantages:")
    st.markdown("\n".join(f"- {disadv}" for disadv in details["Disadvantages"]))
with common_use_cases:
    st.subheader("Common Use Cases:")
    st.markdown("\n".join(f"- {use_case}" for use_case in details["Common Use Cases"]))

st.subheader("Formula:")
st.latex(details["Formula"])

st.subheader("Example Code:")
st.code(details["Example Code"], language="python")
