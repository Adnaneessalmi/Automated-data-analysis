import pandas as pd
import streamlit as st
import os
import tempfile
from fpdf import FPDF
import matplotlib.pyplot as plt
from datetime import datetime
import re
import logging  

# Set up logger
logger = logging.getLogger(__name__)

# Function to load data
def load_data(uploaded_file):
    try:
        # Read the uploaded CSV file into a pandas DataFrame
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error reading the CSV file: {str(e)}")
        logger.error(f"Error reading CSV file: {str(e)}")
        raise

# Function to save a plot to a temporary file
def save_plot_to_temp_file(fig):
    # Create a temporary directory to store the plot image
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, f"temp_plot_{len(os.listdir(temp_dir))}.png")
    
    # Save the figure to the temporary file
    fig.savefig(temp_file_path, format='png')
    plt.close(fig)  # Close the figure to free up memory

    return temp_file_path

# Function to remove HTML tags
def remove_html_tags(text):
    clean_text = re.sub(r'<[^>]+>', '', text)
    return clean_text

# Enhanced PDF Generation Function with Consistent Page Borders
class PDFWithBorder(FPDF):
    def header(self):
        # Add a consistent border around each page
        self.set_line_width(0.5)
        self.rect(5.0, 5.0, 200.0, 287.0)

def export_to_pdf(figure_files, analysis_results):
    pdf = PDFWithBorder()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Adding the first page with a border
    pdf.add_page()

    # Title and subtitle on the first page
    pdf.set_font("Arial", 'B', 28)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(0, 20, "Advanced Data Analysis Report", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", 'I', 16)
    pdf.set_text_color(100, 100, 100)
    current_date = datetime.now().strftime("%B %d, %Y")
    pdf.cell(0, 10, f"Prepared by: Data Analysis Team", ln=True, align='C')
    pdf.cell(0, 10, f"Date of Analysis: {current_date}", ln=True, align='C')
    pdf.ln(15)

    # Adding report overview
    pdf.set_draw_color(0, 51, 102)
    pdf.set_line_width(0.7)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())  # Draw a line for separation
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 18)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(0, 15, "Report Overview", ln=True, align='L')
    pdf.ln(5)
    
    pdf.set_font("Arial", size=12)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(0, 10, "This report presents a comprehensive analysis of the provided dataset, "
                           "highlighting key insights from various types of analyses including "
                           "distribution analysis, correlation analysis, PCA, clustering, and time-series analysis. "
                           "The visualizations and metrics are presented with a focus on data-driven decision making.")
    pdf.ln(10)

    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, "Let's dive into the details!", ln=True, align='C')
    pdf.ln(10)

    # Analysis Results Overview on the same page if possible
    pdf.set_font("Arial", 'B', 16)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(0, 10, "Analysis Results Overview", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)

    # Adding analysis results without adding extra pages unnecessarily
    for result in analysis_results.split("\n"):
        clean_result = remove_html_tags(result)
        if clean_result.strip() and clean_result.strip() != "-":
            pdf.multi_cell(0, 10, f"- {clean_result}")
    pdf.ln(10)

    # Adding figures with titles
    for i, fig_path in enumerate(figure_files):
        pdf.add_page()

        pdf.set_font("Arial", 'B', 14)
        pdf.set_text_color(0, 51, 102)
        pdf.cell(0, 10, f"Figure {i+1}: Analysis Visualization", ln=True)
        pdf.ln(5)
        pdf.image(fig_path, x=10, y=None, w=180)  # Adjust the width to fit the page while maintaining aspect ratio
        pdf.ln(10)

    # Adding a final summary page
    pdf.add_page()

    pdf.set_font("Arial", 'B', 16)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(0, 10, "Summary and Next Steps", ln=True)
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(0, 10, "This report provides detailed visualizations and analyses, which are crucial for data-driven decision-making. "
                           "Based on these results, the following actions are recommended:\n\n"
                           "- Evaluate key trends from the time-series analysis.\n"
                           "- Investigate areas with strong correlations.\n"
                           "- Use the clustering results to segment data for targeted strategies.\n\n"
                           "We hope this report provides value and aids in informed decision-making.")

    # Return the PDF as a binary output
    return pdf.output(dest='S').encode('latin1')
