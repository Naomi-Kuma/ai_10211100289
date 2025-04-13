import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def run():
    st.title("📈 Regression Analysis Tool")


# Back button
    if st.button("⬅️ Back to Services"):
        st.session_state.page = "services"
        st.rerun()

    st.markdown("""
        <style>
            .stButton > button {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 0.5em 2em;
                font-size: 18px;
                transition: background-color 0.3s ease;
            }
            .stButton > button:hover {
                background-color: #45a049;
            }
        </style>
    """, unsafe_allow_html=True)

        
    # Step 1: Upload CSV
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.subheader("📋 Dataset Preview")
        st.write(data.head())

        # Check if data has at least two columns
        if data.shape[1] < 2:
            st.warning("Please upload a CSV with at least two columns: one for features and one for the target.")
        
        # Select feature and target columns
        feature_columns = data.columns.tolist()
        target_column = st.selectbox("Select Target Column", options=feature_columns)

        feature_columns.remove(target_column)
        selected_features = st.multiselect("Select Feature Columns", options=feature_columns, default=feature_columns)

        if len(selected_features) > 0:
            st.subheader("📑 Selected Features")
            st.write(selected_features)

            # Preprocessing options: Handling missing data and normalization
            st.subheader("🔧 Preprocessing Options")
            handle_missing = st.checkbox("Handle Missing Data", value=True)
            normalize_data = st.checkbox("Normalize Data", value=False)

            # Prepare data
            X = data[selected_features]
            y = data[target_column]

            # Handle missing values if checkbox is checked
            if handle_missing:
                # Handle missing values only in numeric columns
                numeric_cols = X.select_dtypes(include=[np.number]).columns
                X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())  # Fill NaN with mean

            # Normalize data if checkbox is checked
            if normalize_data:
                # Normalize only numeric columns
                numeric_cols = X.select_dtypes(include=[np.number]).columns
                scaler = StandardScaler()
                X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

            # One-Hot Encoding for categorical variables in features
            categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
            numeric_columns = X.select_dtypes(exclude=['object']).columns.tolist()

            # Encode the target variable (if categorical) using LabelEncoder
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)

            # Column transformer: Apply OneHotEncoding to categorical columns
            preprocessor = ColumnTransformer(
                transformers=[
                    ('cat', OneHotEncoder(), categorical_columns),
                    ('num', 'passthrough', numeric_columns)
                ])

            # Train-Test Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Create a pipeline that includes preprocessing and model fitting
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', LinearRegression())
            ])

            # Add a button to trigger the regression
            if st.button("Run Regression"):
                # Train the model
                pipeline.fit(X_train, y_train)

                # Predictions
                y_pred = pipeline.predict(X_test)

                # Evaluate the model
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                # Display results
                st.subheader("📊 Regression Results")
                st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
                st.write(f"Mean Squared Error (MSE): {mse:.2f}")
                st.write(f"R-squared: {r2:.2f}")

                # Scatter Plot: Predicted vs Actual
                st.subheader("🔮 Scatter Plot: Predicted vs Actual")
                fig, ax = plt.subplots()
                ax.scatter(y_test, y_pred)
                ax.set_xlabel("Actual Values")
                ax.set_ylabel("Predicted Values")
                ax.set_title("Scatter Plot: Predicted vs Actual")
                st.pyplot(fig)

                # Input box for custom prediction (only if there's one feature)
                if len(selected_features) == 1:
                    st.subheader("📝 Custom Prediction Input")
                    feature_value = st.number_input(f"Enter value for {selected_features[0]}", min_value=0.0, value=0.0)
                    custom_prediction = pipeline.predict([[feature_value]])
                    st.write(f"Predicted value for {selected_features[0]} = {custom_prediction[0]}")

                # Visualization of regression line (only if there's one feature)
                if len(selected_features) == 1:
                    st.subheader("📉 Regression Line")
                    fig, ax = plt.subplots()
                    ax.scatter(X_test, y_test, color='blue', label='Actual')
                    ax.plot(X_test, pipeline.predict(X_test), color='red', label='Regression Line')
                    ax.set_xlabel(selected_features[0])
                    ax.set_ylabel(target_column)
                    ax.set_title("Regression Line")
                    st.pyplot(fig)

    else:
        st.info("Please upload a CSV file to begin the regression session.")

