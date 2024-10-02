import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer, LabelEncoder
import logging

logger = logging.getLogger(__name__)

def preprocess_data(df, numeric_columns):
    # Split columns with missing values into numerical and categorical
    columns_with_missing_values = df.columns[df.isna().any()]
    numerical_missing = [col for col in columns_with_missing_values if col in numeric_columns]
    categorical_missing = [col for col in columns_with_missing_values if col not in numeric_columns]

    # Handle missing values for numerical columns
    if numerical_missing:
        st.header("Handle Missing Values for Numerical Columns")
        for col in numerical_missing:
            st.write(f"Column '{col}' has {df[col].isna().sum()} missing values.")
            missing_option = st.selectbox(
                f"How would you like to handle missing values for '{col}'?",
                ["Replace with Median", "Replace with Mean", "Extrapolate", "Drop Rows with Missing Values"]
            )
            if missing_option == "Replace with Median":
                df[col].fillna(df[col].median(), inplace=True)
                st.write(f"Missing values in '{col}' have been replaced with the median.")
            elif missing_option == "Replace with Mean":
                df[col].fillna(df[col].mean(), inplace=True)
                st.write(f"Missing values in '{col}' have been replaced with the mean.")
            elif missing_option == "Extrapolate":
                df[col] = df[col].interpolate(method='linear', limit_direction='forward', axis=0)
                st.write(f"Missing values in '{col}' have been extrapolated.")
            elif missing_option == "Drop Rows with Missing Values":
                df.dropna(subset=[col], inplace=True)
                st.write(f"Rows with missing values in '{col}' have been dropped.")
            logger.info(f"Missing values in '{col}' handled using {missing_option}")

    # Handle missing values for categorical columns
    if categorical_missing:
        st.header("Handle Missing Values for Categorical Columns")
        for col in categorical_missing:
            st.write(f"Column '{col}' has {df[col].isna().sum()} missing values.")
            missing_option = st.selectbox(
                f"How would you like to handle missing values for '{col}'?",
                ["Replace with Mode", "Replace with Row After (if it exists)", "Replace with Row Before", "Drop Rows with Missing Values"]
            )
            if missing_option == "Replace with Mode":
                df[col].fillna(df[col].mode()[0], inplace=True)
                st.write(f"Missing values in '{col}' have been replaced with the mode.")
            elif missing_option == "Replace with Row After (if it exists)":
                df[col] = df[col].fillna(method='bfill')
                st.write(f"Missing values in '{col}' have been replaced with the value from the row after.")
            elif missing_option == "Replace with Row Before":
                df[col] = df[col].fillna(method='ffill')
                st.write(f"Missing values in '{col}' have been replaced with the value from the row before.")
            elif missing_option == "Drop Rows with Missing Values":
                df.dropna(subset=[col], inplace=True)
                st.write(f"Rows with missing values in '{col}' have been dropped.")
            logger.info(f"Missing values in '{col}' handled using {missing_option}")

    # Handle categorical columns (encoding)
    categorical_columns = df.select_dtypes(include=['object']).columns
    if st.checkbox("Encode categorical features"):
        for col in categorical_columns:
            df[col] = LabelEncoder().fit_transform(df[col])
        st.write("Categorical features have been encoded.")
        logger.info("Categorical features encoded")

    # Update numeric columns after encoding
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

    # Scaling options
    if st.checkbox("Scale features"):
        st.write("Feature Scaling Options:")
        scaler = st.selectbox("Select scaling method", ["StandardScaler", "MinMaxScaler", "None"])
        if scaler == "StandardScaler":
            df[numeric_columns] = StandardScaler().fit_transform(df[numeric_columns])
            st.write("Features have been scaled using StandardScaler.")
            logger.info("Features scaled using StandardScaler")
        elif scaler == "MinMaxScaler":
            df[numeric_columns] = MinMaxScaler().fit_transform(df[numeric_columns])
            st.write("Features have been scaled using MinMaxScaler.")
            logger.info("Features scaled using MinMaxScaler")

    # Apply transformations
    if st.checkbox("Apply transformations"):
        st.write("Feature Transformation Options:")
        transformation = st.selectbox("Select transformation method", ["Log Transform", "Power Transformer (Yeo-Johnson)", "None"])
        
        if transformation == "Log Transform":
            # Apply log transform only to positive numeric columns
            log_transform_columns = df[numeric_columns].columns[(df[numeric_columns] > 0).all()]
            df[log_transform_columns] = df[log_transform_columns].apply(np.log)
            st.write(f"Log transformation applied to columns: {list(log_transform_columns)}")
            logger.info(f"Log transformation applied to columns: {list(log_transform_columns)}")
        elif transformation == "Power Transformer (Yeo-Johnson)":
            transformer = PowerTransformer(method='yeo-johnson')
            df[numeric_columns] = transformer.fit_transform(df[numeric_columns])
            st.write("Features have been transformed using Yeo-Johnson Power Transformer.")
            logger.info("Features transformed using Yeo-Johnson Power Transformer")

    return df
