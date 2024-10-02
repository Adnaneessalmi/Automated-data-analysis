import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from utils import save_plot_to_temp_file
import logging

logger = logging.getLogger(__name__)

# Time-Series Analysis Function
def perform_time_series_analysis(df, numeric_columns, analysis_results, figure_files, date_column):
    try:
        # Ensure the date column is in datetime format
        if date_column not in df.columns:
            raise ValueError(f"Date column '{date_column}' is not in the dataframe.")

        # Set the date column as the index for time-based analysis
        df = df.set_index(date_column)

        # Perform time-based analysis for selected metrics
        st.subheader("Select Metrics for Time-Based Trend Analysis")
        selected_metrics = st.multiselect("Select metrics for time-based analysis", numeric_columns)

        # Dictionary to store figures for later use in the PDF
        time_series_figures = {}

        # Display and save time-based analysis for each selected metric
        for metric in selected_metrics:
            resampled_data = df[metric].resample('M').mean()

            # Create a new figure for each metric
            fig, ax = plt.subplots()
            ax.plot(resampled_data.index, resampled_data.values)
            ax.set_title(f"Time-Based Analysis for {metric}")
            ax.set_xlabel('Time')
            ax.set_ylabel(metric)
            plt.xticks(rotation=45)
            st.pyplot(fig)

            # Save the figure for later inclusion in PDF
            time_series_figures[metric] = fig

        # After displaying all graphs, ask which ones to add to the PDF report
        st.subheader("Which trends do you want to add in the PDF report?")
        for metric in numeric_columns:
            add_to_pdf = st.checkbox(f"Add time-based analysis of '{metric}' to PDF", value=False)

            # If the metric has been visualized, use the stored figure; otherwise, create a new figure
            if add_to_pdf:
                if metric in time_series_figures:
                    fig = time_series_figures[metric]
                else:
                    # Create a new figure for metrics not previously selected
                    resampled_data = df[metric].resample('M').mean()
                    fig, ax = plt.subplots()
                    ax.plot(resampled_data.index, resampled_data.values)
                    ax.set_title(f"Time-Based Analysis for {metric}")
                    ax.set_xlabel('Time')
                    ax.set_ylabel(metric)
                    plt.xticks(rotation=45)

                # Save the figure for the PDF
                fig_path = save_plot_to_temp_file(fig)
                figure_files.append(fig_path)
                analysis_results.append(f"Performed time-based analysis on {metric} with monthly averages.")
                logger.info(f"Time-based analysis for {metric} added to PDF.")
            else:
                logger.info(f"Time-based analysis for {metric} not added to PDF.")

    except Exception as e:
        logger.error(f"Error in time-based analysis: {str(e)}")
        st.error(f"An error occurred during time-series analysis: {str(e)}")
