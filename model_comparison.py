import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, accuracy_score
import logging

logger = logging.getLogger(__name__)

# Model Comparison Function
def compare_models(df, numeric_columns, analysis_results):
    # Select the target column for model training
    target_column = st.selectbox("Select target column for model comparison", df.columns)
    feature_columns = [col for col in numeric_columns if col != target_column]
    selected_features = st.multiselect("Select features for the model", feature_columns, default=feature_columns)

    if not selected_features:
        st.error("Please select at least one feature for the model.")
        return

    # Prepare data for training
    X = df[selected_features]
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if pd.api.types.is_object_dtype(df[target_column]):
        # Classification Models
        models = {
            "Random Forest Classifier": RandomForestClassifier(n_estimators=100, random_state=42),
            "Decision Tree Classifier": DecisionTreeClassifier(random_state=42)
        }

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            st.write(f"{model_name} Accuracy: {score:.2f}")
            analysis_results.append(f"{model_name} achieved accuracy of {score:.2f}")
            logger.info(f"{model_name} achieved accuracy of {score:.2f}")

        # Allow user to input values for prediction
        st.subheader("Make a Prediction")
        user_input = {}
        for feature in selected_features:
            user_input[feature] = st.text_input(f"Enter value for {feature}", value=str(X[feature].mean()))

        if st.button("Predict"):
            input_data = pd.DataFrame([user_input])
            for col in selected_features:
                input_data[col] = pd.to_numeric(input_data[col], errors='coerce')  # Convert inputs to numeric
            prediction = model.predict(input_data)
            st.write(f"Predicted {target_column}: {prediction[0]}")

    else:
        # Regression Models
        models = {
            "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
            "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42),
            "Linear Regression": LinearRegression()
        }

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = r2_score(y_test, y_pred)
            st.write(f"{model_name} R2 Score: {score:.2f}")
            analysis_results.append(f"{model_name} achieved R2 Score of {score:.2f}")
            logger.info(f"{model_name} achieved R2 Score of {score:.2f}")

        # Allow user to input values for prediction
        st.subheader("Make a Prediction")
        user_input = {}
        for feature in selected_features:
            user_input[feature] = st.number_input(f"Enter value for {feature}", value=float(X[feature].mean()))

        if st.button("Predict"):
            input_data = pd.DataFrame([user_input])
            prediction = model.predict(input_data)
            st.write(f"Predicted {target_column}: {prediction[0]}")
