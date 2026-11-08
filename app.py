import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, ConfusionMatrixDisplay
)

# 1. Page Configuration
st.set_page_config(page_title="Predictive Maintenance Classifier", layout="wide")
st.title("Industrial Machine Predictive Maintenance Dashboard")
st.markdown("Evaluate pre-trained machine learning classifiers on test equipment data.")

# 2. Sidebar Controls
st.sidebar.header("User Controls")

model_paths = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Support Vector Machine": "model/support_vector_machine.pkl"
}

chosen_model_name = st.sidebar.selectbox("Choose Classification Model", list(model_paths.keys()))
uploaded_csv = st.sidebar.file_uploader("Upload test_data.csv", type=["csv"])

TARGET_COLUMN = "Machine failure"

# 3. Main Evaluation Logic
if uploaded_csv is not None:
    test_df = pd.read_csv(uploaded_csv)
    
    st.subheader("Uploaded Test Data Sample")
    st.dataframe(test_df.head(5))

    if TARGET_COLUMN not in test_df.columns:
        st.error(f"Error: Target column '{TARGET_COLUMN}' is missing from the uploaded CSV.")
    else:
        # Separate features from ground truth
        X_test = test_df.drop(columns=[TARGET_COLUMN])
        y_test = test_df[TARGET_COLUMN]

        # Load scaler and transform inputs
        scaler = joblib.load("model/scaler.pkl")
        X_test_scaled = scaler.transform(X_test)

        # Load pre-trained model
        model = joblib.load(model_paths[chosen_model_name])

        # Run inference without retraining
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        # Calculate the 6 evaluation metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)

        st.divider()
        st.subheader(f"Evaluation Metrics: {chosen_model_name}")

        # Display the 6 metric cards
        row1_col1, row1_col2, row1_col3 = st.columns(3)
        row1_col1.metric("Accuracy", f"{acc:.4f}")
        row1_col2.metric("AUC Score", f"{auc:.4f}")
        row1_col3.metric("Precision", f"{prec:.4f}")

        row2_col1, row2_col2, row2_col3 = st.columns(3)
        row2_col1.metric("Recall", f"{rec:.4f}")
        row2_col2.metric("F1 Score", f"{f1:.4f}")
        row2_col3.metric("MCC Score", f"{mcc:.4f}")

        st.divider()

        # Display Confusion Matrix
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(5, 4))
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Failure (0)", "Failure (1)"])
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        st.pyplot(fig)
else:
    st.info("Please upload 'test_data.csv' using the sidebar to view metrics and confusion matrix.")