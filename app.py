# anomaly_detection_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

import shap
from scipy import stats

# Function to calculate Z-Score anomalies
def z_score_anomaly_detection(df, feature):
    z_scores = np.abs(stats.zscore(df[feature]))
    anomaly_threshold = 3
    df['z_score_anomaly'] = (z_scores > anomaly_threshold).astype(int)
    return df

# Function to train Isolation Forest
def train_isolation_forest(X_train):
    model = IsolationForest(n_estimators=100, contamination=0.04, random_state=42)
    model.fit(X_train)
    return model

# SHAP explanation
def explain_with_shap(model, X_test):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    st.subheader("SHAP Feature Importance")
    # Create the figure and plot SHAP values
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    st.pyplot(fig)

# Sample Data Creation
@st.cache_data
def generate_sample_data():
    np.random.seed(42)
    num_records = 1000
    hosts = [f'host_{i}' for i in range(1, 6)]
    buckets = [f'bucket_{i}' for i in range(1, 4)]
    prefixes = ['images/', 'videos/', 'documents/']
    methods = ['GET', 'POST', 'HEAD']
    responses = [200, 404, 500]

    data = {
        'host': np.random.choice(hosts, num_records),
        'bucket': np.random.choice(buckets, num_records),
        'prefix': np.random.choice(prefixes, num_records),
        'request_method': np.random.choice(methods, num_records),
        'request_date': pd.date_range(start='2024-01-01', periods=num_records, freq='H'),
        'response': np.random.choice(responses, num_records, p=[0.7, 0.2, 0.1]),
        'latency': np.random.normal(loc=100, scale=20, size=num_records)
    }
    df = pd.DataFrame(data)
    
    # Introduce anomalies
    anomaly_periods = np.random.choice(df.index, size=int(0.05 * num_records), replace=False)
    df.loc[anomaly_periods, 'latency'] += 100  # artificially increase latency for anomalies
    
    return df

# Streamlit App
def main():
    st.title("Anomaly Detection in Object Storage Logs")

    # Step 1: Upload or generate data
    st.header("1. Upload or Generate Dataset")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
    else:
        st.info("Using generated sample data.")
        df = generate_sample_data()

    # Step 2: Data preview
    st.header("2. Data Preview")
    st.write(df.head())

    # Feature engineering
    df['hour'] = df['request_date'].dt.hour
    df['day_of_week'] = df['request_date'].dt.dayofweek

    # Step 3: Choose whether to scale features
    st.header("3. Data Preprocessing Options")
    scale_option = st.selectbox("Would you like to scale the features?", ("Yes", "No"))
    
    if scale_option == "Yes":
        numerical_features = ['latency', 'hour', 'day_of_week']
        scaler = StandardScaler()
        df[numerical_features] = scaler.fit_transform(df[numerical_features])
        st.success("Features scaled successfully!")
        st.write(df[numerical_features].head())
    else:
        st.info("No scaling applied.")

    # Select features for training
    feature_columns = ['latency', 'hour', 'day_of_week']
    X = df[feature_columns]

    # Step 4: Model Selection
    st.header("4. Choose Anomaly Detection Method")
    model_choice = st.selectbox("Select a model", ("Z-Score (Statistical)", "Isolation Forest (Machine Learning)"))

    # Explanation for Z-Score
    if model_choice == "Z-Score (Statistical)":
        st.subheader("What is Z-Score and How Does It Work?")
        st.write("""
        **Z-Score** is a statistical measure that tells us how far a data point is from the mean of the dataset in terms of standard deviations.
        
        For this use case, the Z-Score method is used to identify anomalies in the `latency` feature:
        - The Z-Score for each latency value is calculated.
        - If the Z-Score exceeds a threshold (typically 3), the data point is flagged as an anomaly.
        
        **Why Use Z-Score for Anomalies?**
        - In object storage logs, unusually high latency can indicate performance issues such as network congestion, server overload, or configuration problems.
        - The Z-Score method effectively flags these outliers because it measures how far the latency deviates from the average.
        """)

    # Explanation for Isolation Forest
    elif model_choice == "Isolation Forest (Machine Learning)":
        st.subheader("What is Isolation Forest and How Does It Work?")
        st.write("""
        **Isolation Forest** is a machine learning model that isolates anomalies by randomly selecting features and splitting the data.
        
        For this use case, the Isolation Forest model is trained on the features `latency`, `hour`, and `day_of_week`. The model isolates anomalies based on how different these features are from normal patterns:
        - Isolation Forest creates multiple random splits in the data.
        - Anomalies are isolated quickly because they differ significantly from the majority of data points.
        - The model predicts whether each data point is normal or an anomaly.
        
        **Why Use Isolation Forest for Anomalies?**
        - In object storage logs, patterns such as high latency during unusual hours or on specific days may indicate anomalies.
        - The Isolation Forest can detect these anomalies based on multiple factors (like latency, time of day, and day of the week).
        """)

    if st.button("Select your Approach - Stats vs Model"):
        if model_choice == "Z-Score (Statistical)":
            st.subheader("Anomaly Detection Using Z-Score")
            df = z_score_anomaly_detection(df, 'latency')
            st.write(df[['request_date', 'latency', 'z_score_anomaly']].head())
            st.success("Z-Score anomalies detected.")

            # Plotting anomalies over time (time series)
            st.subheader("Anomalies Detected Over Time (Z-Score)")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(df['request_date'], df['latency'], label="Latency")
            ax.scatter(df['request_date'][df['z_score_anomaly'] == 1], df['latency'][df['z_score_anomaly'] == 1], color='red', label="Anomaly", zorder=5)
            ax.set_title("Latency Time Series with Anomalies")
            ax.set_xlabel("Request Date")
            ax.set_ylabel("Latency")
            ax.legend()
            st.pyplot(fig)

            st.subheader("Detected Anomalies and Explanations (Z-Score)")
            anomaly_details = df[df['z_score_anomaly'] == 1][['request_date', 'latency', 'hour', 'day_of_week']]
            st.write(anomaly_details)
            st.write("**Possible Explanations:**")
            st.write("""
            - High latency values could indicate server overload, slow network response, or inefficient processing.
            - Anomalies could occur during specific hours or on certain days of the week when the system is under heavy load.
            """)

        elif model_choice == "Isolation Forest (Machine Learning)":
            st.subheader("Train Isolation Forest Model")
            model = train_isolation_forest(X)
            st.success("Isolation Forest Model Trained!")

            st.subheader("Predict Anomalies with Isolation Forest")
            df['iso_forest_anomaly'] = model.predict(X)
            df['iso_forest_anomaly'] = np.where(df['iso_forest_anomaly'] == -1, 1, 0)
            st.write(df[['request_date', 'latency', 'iso_forest_anomaly']].head())

            # Plotting anomalies over time (time series)
            st.subheader("Anomalies Detected Over Time (Isolation Forest)")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(df['request_date'], df['latency'], label="Latency")
            ax.scatter(df['request_date'][df['iso_forest_anomaly'] == 1], df['latency'][df['iso_forest_anomaly'] == 1], color='red', label="Anomaly", zorder=5)
            ax.set_title("Latency Time Series with Anomalies")
            ax.set_xlabel("Request Date")
            ax.set_ylabel("Latency")
            ax.legend()
            st.pyplot(fig)

            st.subheader("Detected Anomalies and Explanations (Isolation Forest)")
            anomaly_details = df[df['iso_forest_anomaly'] == 1][['request_date', 'latency', 'hour', 'day_of_week']]
            st.write(anomaly_details)
            st.write("**Possible Explanations:**")
            st.write("""
            - The Isolation Forest algorithm flagged these points as anomalies because they differ significantly from normal behavior.
            - High latency or unusual access patterns (e.g., during off-peak hours) could indicate misconfigurations, network issues, or external attacks.
            """)

            # Model Evaluation
            st.header("Model Evaluation")
            z_score_anomalies = z_score_anomaly_detection(df, 'latency')['z_score_anomaly']
            st.subheader("Confusion Matrix")
            st.write(confusion_matrix(z_score_anomalies, df['iso_forest_anomaly']))

            st.subheader("Classification Report")
            st.text(classification_report(z_score_anomalies, df['iso_forest_anomaly']))

            st.subheader("ROC-AUC Score")
            auc_score = roc_auc_score(z_score_anomalies, df['iso_forest_anomaly'])
            st.write(f"ROC-AUC Score: {auc_score:.4f}")

            # Explain model predictions using SHAP
            st.header("Model Explainability using SHAP")
            explain_with_shap(model, X)

    # Short note on other anomaly detection techniques
    st.header("Other Anomaly Detection Techniques (Short Notes)")
    st.write("""
    1. **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**:
        - A clustering algorithm that detects anomalies as points that do not belong to any cluster.
        - Works well when anomalies are separated from normal points in a high-dimensional space.

    2. **One-Class SVM**:
        - A variation of Support Vector Machines (SVM) used for anomaly detection.
        - Learns a boundary around the normal data points, and points outside the boundary are considered anomalies.

    3. **Autoencoders (Neural Networks)**:
        - An unsupervised deep learning technique that reconstructs input data.
        - Anomalies are detected based on the reconstruction error, where high errors indicate anomalous points.
    """)

if __name__ == "__main__":
    main()
