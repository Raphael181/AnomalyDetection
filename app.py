import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import io

# Load model and scaler
model = joblib.load('isolation_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

def generate_synthetic_data(n_samples=10000, n_anomalies=50, n_outliers=10):
    np.random.seed(42)
    date_range = pd.date_range(start='2024-01-01', periods=n_samples, freq='H')
    normal_data = np.random.normal(loc=50, scale=10, size=n_samples)
    anomaly_indices = np.random.choice(n_samples, size=n_anomalies, replace=False)
    normal_data[anomaly_indices] += np.random.uniform(50, 100, size=n_anomalies)
    outlier_indices = np.random.choice(n_samples, size=n_outliers, replace=False)
    normal_data[outlier_indices] = np.random.uniform(200, 300, size=n_outliers)
    df = pd.DataFrame({
        'datetime': date_range,
        'Global_active_power': normal_data
    })
    return df

def plot_data(df):
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(df['datetime'], df['Global_active_power'], color='blue', label='Energy Consumption', linewidth=1.5)
    ax.scatter(df[df['anomaly'] == -1]['datetime'], df[df['anomaly'] == -1]['Global_active_power'], color='red', label='Anomalies', marker='o')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Global Active Power', fontsize=12)
    ax.set_title('Anomaly Detection Results', fontsize=16)
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def main():
    st.title('Anomaly Detection in Energy Consumption Data')
    st.write("This application demonstrates anomaly detection in energy consumption data using the Isolation Forest algorithm.")

    st.sidebar.header('Settings')
    n_samples = st.sidebar.slider('Number of Samples', min_value=1000, max_value=20000, value=10000, step=1000)
    n_anomalies = st.sidebar.slider('Number of Anomalies', min_value=10, max_value=100, value=50, step=10)
    n_outliers = st.sidebar.slider('Number of Outliers', min_value=5, max_value=50, value=10, step=5)

    # Generate synthetic data with user-defined settings
    df = generate_synthetic_data(n_samples, n_anomalies, n_outliers)
    X = df[['Global_active_power']].values
    X_scaled = scaler.transform(X)
    df['anomaly'] = model.predict(X_scaled)

    # Display the data
    st.subheader('Data Preview')
    st.write(df.head())

    # Plot the results
    st.subheader('Anomaly Detection Visualization')
    fig = plot_data(df)
    st.pyplot(fig)

    # Additional Information
    st.sidebar.markdown("### About")
    st.sidebar.markdown("""
    This application uses a synthetic dataset to demonstrate anomaly detection using the Isolation Forest model. 
    You can adjust the number of samples, anomalies, and outliers using the sliders in the sidebar.
    """)

if __name__ == "__main__":
    main()
