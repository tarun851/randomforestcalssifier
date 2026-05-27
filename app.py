import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Random Forest Classifier",
    page_icon="🌲",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
    color: white;
}

h1, h2, h3 {
    color: #00F5FF;
}

.stButton>button {
    background-color: #00F5FF;
    color: black;
    border-radius: 10px;
    border: none;
    padding: 10px 20px;
    font-weight: bold;
}

div[data-testid="stMetric"] {
    background-color: #1C1F26;
    padding: 15px;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================

st.title("🌲 Random Forest Classification")

st.markdown("---")

# =========================================================
# LOAD DATASET
# =========================================================

data = load_breast_cancer()

X = data.data
y = data.target

features = data.feature_names

df = pd.DataFrame(X, columns=features)

df['target'] = y

# =========================================================
# DATA PREVIEW
# =========================================================

st.subheader("📊 Dataset Preview")

st.dataframe(df.head(), use_container_width=True)

# =========================================================
# SHAPE
# =========================================================

st.subheader("📐 Dataset Shape")

st.write(df.shape)

# =========================================================
# VISUALIZATION
# =========================================================

st.subheader("📈 Visualizations")

v1, v2 = st.columns(2)

# -------------------- TARGET DISTRIBUTION --------------------

with v1:

    fig1, ax1 = plt.subplots(figsize=(5,4))

    df['target'].value_counts().plot(
        kind='bar',
        ax=ax1
    )

    ax1.set_title("Target Distribution")

    st.pyplot(fig1)

# -------------------- FEATURE CORRELATION --------------------

with v2:

    corr = df.corr()['target'].sort_values()

    fig2, ax2 = plt.subplots(figsize=(5,4))

    corr.plot(
        kind='barh',
        ax=ax2
    )

    ax2.set_title("Feature Correlation")

    st.pyplot(fig2)

# =========================================================
# HEATMAP
# =========================================================

st.subheader("🔥 Correlation Heatmap")

fig3, ax3 = plt.subplots(figsize=(12,8))

sns.heatmap(
    df.corr(),
    cmap='coolwarm',
    linewidths=0.5,
    ax=ax3
)

st.pyplot(fig3)

# =========================================================
# REMOVE OUTLIERS
# =========================================================

cleaned_df = df.copy()

for col in cleaned_df.columns[:-1]:

    Q1 = cleaned_df[col].quantile(0.25)

    Q3 = cleaned_df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR

    upper = Q3 + 1.5 * IQR

    cleaned_df = cleaned_df[
        (cleaned_df[col] >= lower) &
        (cleaned_df[col] <= upper)
    ]

# =========================================================
# CLEANED DATA SHAPE
# =========================================================

st.subheader("🧹 Dataset Shape After Outlier Removal")

st.write(cleaned_df.shape)

# =========================================================
# FEATURES & TARGET
# =========================================================

X = cleaned_df.drop('target', axis=1)

y = cleaned_df['target']

# =========================================================
# SIDEBAR HYPERPARAMETERS
# =========================================================

st.sidebar.header("⚙️ Hyperparameter Selection")

n_estimators = st.sidebar.multiselect(
    "Select n_estimators",
    [50, 100, 200, 300, 500],
    default=[100, 200]
)

max_depth = st.sidebar.multiselect(
    "Select max_depth",
    [None, 5, 7, 9, 11, 15],
    default=[5, 7, 9]
)

min_samples_split = st.sidebar.multiselect(
    "Select min_samples_split",
    [2, 4, 5, 10],
    default=[2, 4]
)

min_samples_leaf = st.sidebar.multiselect(
    "Select min_samples_leaf",
    [1, 2, 4, 6],
    default=[1, 2]
)

test_size = st.sidebar.slider(
    "Test Size",
    0.1,
    0.4,
    0.2
)

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=42
)

# =========================================================
# MODEL
# =========================================================

model = RandomForestClassifier(random_state=42)

# =========================================================
# PARAM GRID
# =========================================================

params = {

    'n_estimators': n_estimators,

    'max_depth': max_depth,

    'min_samples_split': min_samples_split,

    'min_samples_leaf': min_samples_leaf
}

# =========================================================
# GRID SEARCH
# =========================================================

with st.spinner("Finding Best Hyperparameters..."):

    grid = GridSearchCV(
        estimator=model,
        param_grid=params,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

# =========================================================
# BEST MODEL
# =========================================================

best_model = grid.best_estimator_

# =========================================================
# SAVE MODEL
# =========================================================

os.makedirs("models", exist_ok=True)

joblib.dump(
    best_model,
    "models/random_forest_model.pkl"
)

# =========================================================
# PREDICTIONS
# =========================================================

y_pred = best_model.predict(X_test)

# =========================================================
# ACCURACY
# =========================================================

accuracy = accuracy_score(y_test, y_pred)

# =========================================================
# METRICS
# =========================================================

st.subheader("📊 Model Performance")

m1, m2, m3 = st.columns(3)

with m1:
    st.metric(
        "Accuracy",
        round(accuracy, 4)
    )

with m2:
    st.metric(
        "Correct Predictions",
        (y_pred == y_test).sum()
    )

with m3:
    st.metric(
        "Wrong Predictions",
        (y_pred != y_test).sum()
    )

# =========================================================
# BEST PARAMETERS
# =========================================================

st.subheader("🏆 Best Hyperparameters")

st.write(grid.best_params_)

# =========================================================
# CONFUSION MATRIX
# =========================================================

st.subheader("📉 Confusion Matrix")

fig4, ax4 = plt.subplots(figsize=(5,4))

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    ax=ax4
)

ax4.set_xlabel("Predicted")

ax4.set_ylabel("Actual")

st.pyplot(fig4)

# =========================================================
# CLASSIFICATION REPORT
# =========================================================

st.subheader("🧠 Classification Report")

report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(report_df, use_container_width=True)

# =========================================================
# FEATURE IMPORTANCE
# =========================================================

st.subheader("⭐ Top Feature Importance")

importance = pd.DataFrame({

    'Feature': X.columns,

    'Importance': best_model.feature_importances_

})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

fig5, ax5 = plt.subplots(figsize=(8,6))

ax5.barh(
    importance['Feature'][:10],
    importance['Importance'][:10]
)

ax5.invert_yaxis()

ax5.set_title("Top 10 Important Features")

st.pyplot(fig5)

# =========================================================
# USER INPUT PREDICTION
# =========================================================

st.markdown("---")

st.subheader("🩺 Predict Cancer Type")

selected_features = [
    'mean radius',
    'mean texture',
    'mean perimeter',
    'mean area',
    'mean smoothness'
]

input_values = []

c1, c2 = st.columns(2)

for idx, feature in enumerate(selected_features):

    min_val = float(df[feature].min())

    max_val = float(df[feature].max())

    mean_val = float(df[feature].mean())

    with c1 if idx % 2 == 0 else c2:

        value = st.slider(
            feature,
            min_val,
            max_val,
            mean_val
        )

    input_values.append(value)

# =========================================================
# TRAIN SMALL MODEL
# =========================================================

X_small = cleaned_df[selected_features]

y_small = cleaned_df['target']

model_small = RandomForestClassifier(
    **grid.best_params_,
    random_state=42
)

model_small.fit(X_small, y_small)

# =========================================================
# INPUT DATAFRAME
# =========================================================

input_df = pd.DataFrame(
    [input_values],
    columns=selected_features
)

# =========================================================
# PREDICTION
# =========================================================

prediction = model_small.predict(input_df)

result = (
    "🟢 Benign (Non-Cancerous)"
    if prediction[0] == 1
    else "🔴 Malignant (Cancerous)"
)

# =========================================================
# BUTTON
# =========================================================

if st.button("Predict"):

    st.subheader("Prediction Result")

    if prediction[0] == 1:

        st.success(result)

    else:

        st.error(result)

# =========================================================
# MODEL SAVED MESSAGE
# =========================================================

st.success(
    "✅ Best model saved inside models/random_forest_model.pkl"
)

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.info("Random Forest Classification App")