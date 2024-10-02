import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
import streamlit as st

# KMeans Clustering Analysis Function
def clustering_analysis(df, numeric_columns):
    # Use KMeans clustering and plot the results explicitly with Matplotlib
    if len(numeric_columns) < 2:
        st.error("At least two numeric columns are required for KMeans clustering.")
        return None

    kmeans = KMeans(n_clusters=3, random_state=42)
    df_cluster = df[numeric_columns].dropna()
    kmeans.fit(df_cluster)

    # Add the cluster labels to the dataframe for plotting
    df_cluster['Cluster'] = kmeans.labels_

    # Create a scatter plot for the first two numeric columns
    fig, ax = plt.subplots()
    scatter = ax.scatter(df_cluster[numeric_columns[0]], df_cluster[numeric_columns[1]], c=df_cluster['Cluster'], cmap='viridis')
    ax.set_title('KMeans Clustering')
    ax.set_xlabel(numeric_columns[0])
    ax.set_ylabel(numeric_columns[1])
    plt.colorbar(scatter)
    
    return fig
