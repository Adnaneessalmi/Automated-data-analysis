import pandas as pd
import streamlit as st
import logging
import matplotlib.pyplot as plt
from preprocessing import preprocess_data
from analysis import distribution_analysis, correlation_analysis, pca_analysis, clustering_analysis
from model_comparison import compare_models
from time_series_analysis import perform_time_series_analysis
from utils import save_plot_to_temp_file, export_to_pdf, load_data

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    st.title("Advanced Analysis Dashboard")

    uploaded_file = st.file_uploader("Upload your CSV data file", type="csv")
    if uploaded_file is not None:
        try:
            # Load data
            df = load_data(uploaded_file)
            logger.info("Data loaded successfully")

            st.write("Data Preview:")
            st.write(df.head())

            # Display missing values before any handling
            st.write("Missing Values in Each Column Before Handling:")
            missing_values = df.isna().sum()
            columns_with_missing_values = missing_values[missing_values > 0]

            if columns_with_missing_values.empty:
                st.write("There are no columns with missing values.")
            else:
                for col, count in columns_with_missing_values.items():
                    st.write(f"Column '{col}': {count} missing values")

            # Initialize variables to None
            id_column = None
            date_column = None

            # Ask the user if there is an ID column to exclude from further analysis
            if st.checkbox("Do you have an identifier (ID) column to exclude from further analysis?"):
                id_column = st.selectbox("Select the ID column", df.columns)
                if id_column:
                    # Drop the selected ID column from the dataframe
                    df = df.drop(columns=[id_column])
                    st.write(f"'{id_column}' column has been excluded from further analysis.")
                    logger.info(f"'{id_column}' column has been excluded from further analysis.")

            # Let the user select a date column and specify the format for proper parsing
            if st.checkbox("Do you have a Date column?"):
                date_column = st.selectbox("Select the Date column", df.columns)
                if date_column:
                    date_format_options = [
                        ("%Y-%m-%d", "Year-Month-Day (e.g., 2024-09-28)"),
                        ("%d-%m-%Y", "Day-Month-Year (e.g., 28-09-2024)"),
                        ("%m-%d-%Y", "Month-Day-Year (e.g., 09-28-2024)"),
                        ("%d/%m/%Y", "Day/Month/Year (e.g., 28/09/2024)"),
                        ("%m/%d/%Y", "Month/Day/Year (e.g., 09/28/2024)"),
                        ("%Y/%m/%d", "Year/Month/Day (e.g., 2024/09/28)"),
                    ]
                    date_format = st.selectbox(
                        "Please select the format that matches your date column:",
                        date_format_options,
                        format_func=lambda x: x[1]
                    )[0]  # Get the actual format string

                    # Convert the date column to datetime format using the specified format
                    try:
                        df[date_column] = pd.to_datetime(df[date_column], format=date_format, errors='coerce')
                        df = df[df[date_column].notna()]  # Drop rows with invalid date values
                        st.write(f"Date column '{date_column}' selected and parsed successfully.")
                        logger.info(f"Date column '{date_column}' selected and parsed successfully.")
                    except Exception:
                        st.error("Failed to parse the date column with the chosen format. Please select a different format.")
                        logger.error(f"Date parsing failed for column '{date_column}' with format '{date_format}'.")
                        return  # Stop further processing if date parsing fails

            # Preprocessing, excluding the ID and date columns
            columns_to_exclude = set()
            if id_column:
                columns_to_exclude.add(id_column)
            if date_column:
                columns_to_exclude.add(date_column)

            columns_to_preprocess = [col for col in df.columns if col not in columns_to_exclude]

            if columns_to_preprocess:
                st.header("Data Preprocessing")
                numeric_columns = df[columns_to_preprocess].select_dtypes(include=['float64', 'int64']).columns
                df_preprocessed = preprocess_data(df[columns_to_preprocess], numeric_columns)
                df = pd.concat([df.drop(columns=columns_to_preprocess), df_preprocessed], axis=1)
                logger.info("Data preprocessing applied successfully.")
            else:
                st.write("No columns available for preprocessing.")

            # Distribution Analysis
            st.header("Distribution Analysis")
            if columns_to_preprocess:
                metric = st.selectbox("Select Metric for Analysis", columns_to_preprocess, index=0)
                fig = distribution_analysis(df, metric)
                st.pyplot(fig)

                # Option to Add Distributions of All Columns to PDF
                st.subheader("What distributions would you like to add in the PDF report?")
                add_to_pdf_columns = {}
                columns_seen = set()

                for column in columns_to_preprocess:
                    if column not in columns_seen:
                        add_to_pdf_columns[column] = st.checkbox(f"Add distribution analysis of {column} to PDF", key=f"dist_{column}")
                        columns_seen.add(column)

                # Add selected figures to the PDF report
                figure_files = []
                analysis_results = []

                for column, add_to_pdf in add_to_pdf_columns.items():
                    if add_to_pdf:
                        fig = distribution_analysis(df, column)
                        figure_files.append(save_plot_to_temp_file(fig))
                        analysis_results.append(f"Analyzed distribution of {column}")
                        logger.info(f"Distribution analysis for {column} added to PDF")

            # Correlation Analysis
            st.header("Correlation Analysis")
            numeric_df = df.select_dtypes(include=['float64', 'int64'])
            if not numeric_df.empty:
                fig = correlation_analysis(numeric_df, numeric_df.columns)
                st.pyplot(fig)
                figure_files.append(save_plot_to_temp_file(fig))
                analysis_results.append("Correlation analysis performed.")
                logger.info("Correlation analysis completed.")
            else:
                st.write("No numeric columns available for correlation analysis.")

            # PCA Analysis
            st.header("PCA Visualization")
            if not numeric_df.empty:
                fig = pca_analysis(df, numeric_df.columns)
                st.pyplot(fig)
                figure_files.append(save_plot_to_temp_file(fig))
                analysis_results.append("PCA analysis performed.")
                logger.info("PCA analysis completed.")
            else:
                st.write("No numeric columns available for PCA analysis.")

            # Clustering Analysis
            st.header("K-Means Clustering")
            if not numeric_df.empty:
                fig = clustering_analysis(df[numeric_df.columns], numeric_df.columns)
                st.pyplot(fig)
                figure_files.append(save_plot_to_temp_file(fig))
                analysis_results.append("Clustering analysis performed.")
                logger.info("Clustering analysis completed.")
            else:
                st.write("No numeric columns available for clustering analysis.")

            # Time-Series Analysis
            st.header("Time-Series Analysis")
            if date_column and st.checkbox("Perform Time-Series Analysis"):
                perform_time_series_analysis(df, numeric_df.columns, analysis_results, figure_files, date_column)

            # Model Comparison
            st.header("Model Comparison")
            columns_for_model_comparison = [col for col in numeric_df.columns if col != date_column]
            compare_models(df[columns_for_model_comparison], columns_for_model_comparison, analysis_results)

            # PDF Export
            if st.button('Generate PDF Report'):
                with st.spinner('Generating PDF Report...'):
                    try:
                        logger.info("PDF generation initiated")
                        pdf_output = export_to_pdf(figure_files, "\n".join(analysis_results))
                        logger.info("PDF generated successfully")
                        st.success('PDF Report generated successfully!')
                        st.download_button(
                            label="Download PDF Report",
                            data=pdf_output,
                            file_name="advanced_analysis_report.pdf",
                            mime="application/pdf"
                        )
                    except Exception as e:
                        logger.error(f"Error in PDF generation: {str(e)}")
                        st.error(f'An error occurred during PDF generation: {str(e)}')

            # Allow user to download the processed CSV file
            processed_filename = uploaded_file.name.split('.')[0] + "_processed.csv"
            st.download_button(
                label="Download Processed CSV File",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name=processed_filename,
                mime='text/csv'
            )

        except Exception as e:
            logger.error(f"An unexpected error occurred: {str(e)}")
            st.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()
