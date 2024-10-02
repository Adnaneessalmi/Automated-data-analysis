import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import logging
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Distribution Analysis
def distribution_analysis(df, metric):
    fig, ax = plt.subplots()
    sns.histplot(df[metric], ax=ax)
    logger.info(f"Distribution analysis for {metric} completed")
    return fig

# Correlation Analysis
def correlation_analysis(df, numeric_columns):
    corr = df[numeric_columns].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    logger.info("Correlation analysis completed")
    return fig

# PCA Analysis
def pca_analysis(df, numeric_columns, n_components=3):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[numeric_columns])
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(scaled_data)
    pca_df = pd.DataFrame(data=pca_result, columns=[f'PC{i+1}' for i in range(n_components)])

    fig, ax = plt.subplots()
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', ax=ax)
    ax.set_title('PCA Visualization')
    logger.info(f"PCA analysis with {n_components} components completed")
    return fig

# Clustering Analysis
def clustering_analysis(df, numeric_columns):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[numeric_columns])
    
    n_clusters = st.slider("Select number of clusters", min_value=2, max_value=10, value=3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_data)
    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)
    pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = cluster_labels

    fig, ax = plt.subplots()
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Cluster', palette='deep', ax=ax)
    ax.set_title(f'K-Means Clustering Visualization ({n_clusters} clusters)')
    logger.info(f"K-means clustering with {n_clusters} clusters completed")
    return fig


# Anomaly Detection Function
def anomaly_detection(df, numeric_columns, analysis_results):
    # Use IsolationForest for anomaly detection
    model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    df_numeric = df[numeric_columns].dropna()  # Drop rows with NaN values for the analysis
    predictions = model.fit_predict(df_numeric)

    # Add anomaly labels to the dataframe
    df_numeric['Anomaly'] = predictions
    df_numeric['Anomaly'] = df_numeric['Anomaly'].apply(lambda x: 'Normal' if x == 1 else 'Anomaly')

    # Count anomalies and normal data points
    anomaly_count = df_numeric[df_numeric['Anomaly'] == 'Anomaly'].shape[0]
    normal_count = df_numeric[df_numeric['Anomaly'] == 'Normal'].shape[0]

    # Record the results
    analysis_results.append(f"Anomaly detection completed with {anomaly_count} anomalies and {normal_count} normal points.")

    return df_numeric