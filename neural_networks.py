import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'home'

def run():
    st.title("🧠 Neural Network for Classification")

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


    # Step 1: Upload Dataset
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df = df.dropna()  # Drop missing values for simplicity
        st.subheader("📋 Dataset Preview")
        st.write(df.head())

        # Step 2: Select target column
        target_col = st.selectbox("Select the target column for classification", df.columns)

        # Step 3: Prepare the data
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Encode labels if they are categorical
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)

        # Apply feature scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Convert X_scaled to DataFrame to retain column names
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Step 4: Build the Feedforward Neural Network
        model = Sequential()
        model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
        model.add(Dense(32, activation='relu'))

        # Determine number of unique classes
        num_classes = len(np.unique(y))

        # Adjust the output layer and loss function based on number of classes
        if num_classes == 2:  # Binary classification
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
        else:  # Multi-class classification
            model.add(Dense(num_classes, activation='softmax'))
            model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

        # Step 5: Hyperparameters - Epochs and Learning Rate
        epochs = st.slider("Number of epochs", min_value=1, max_value=100, value=10)
        learning_rate = st.slider("Learning rate", min_value=0.0001, max_value=0.01, value=0.001)

        # Re-compile model with selected learning rate
        model.compile(loss='binary_crossentropy' if num_classes == 2 else 'sparse_categorical_crossentropy',
                      optimizer=Adam(learning_rate=learning_rate),
                      metrics=['accuracy'])

        # Step 6: Train the model
        st.subheader("Training Progress")
        with st.spinner('Training the model...'):
            history = model.fit(X_train, y_train, epochs=epochs, batch_size=32,
                                validation_data=(X_val, y_val), verbose=0)

        # Step 7: Display Training and Validation Accuracy/Loss
        st.subheader("Training and Validation Accuracy/Loss")
        # Plot the loss and accuracy
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Accuracy
        axes[0].plot(history.history['accuracy'], label='Train Accuracy')
        axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0].set_title("Accuracy")
        axes[0].set_xlabel("Epochs")
        axes[0].set_ylabel("Accuracy")
        axes[0].legend()

        # Loss
        axes[1].plot(history.history['loss'], label='Train Loss')
        axes[1].plot(history.history['val_loss'], label='Validation Loss')
        axes[1].set_title("Loss")
        axes[1].set_xlabel("Epochs")
        axes[1].set_ylabel("Loss")
        axes[1].legend()

        st.pyplot(fig)

        st.success("✅ Model Training Completed!")

        # Step 8: Save the trained model
        save_model = st.button("Save Trained Model")
        if save_model:
            model.save("trained_model.h5")
            st.success("✅ Model saved successfully!")

        # Step 9: Load pre-trained model (if any)
        uploaded_model = st.file_uploader("Upload a pre-trained model", type=["h5"])
        if uploaded_model is not None:
            model = tf.keras.models.load_model(uploaded_model)
            st.success("✅ Pre-trained model loaded successfully!")

        # Step 10: Provide Option for Predictions
        st.subheader("Make Predictions on New Data")

        # Upload test samples
        uploaded_test_file = st.file_uploader("Upload a CSV file with new test samples", type=["csv"])

        if uploaded_test_file is not None:
            test_df = pd.read_csv(uploaded_test_file)
            st.subheader("Test Samples Preview")
            st.write(test_df.head())

            # Prepare the test data
            X_test = test_df.drop(columns=[target_col], errors='ignore')  # Drop target if it exists in the test data
            X_test = pd.DataFrame(X_test, columns=X_train.columns)  # Ensure test columns match training columns
            X_test_scaled = scaler.transform(X_test)  # Scale the test data

            predictions = model.predict(X_test_scaled)

            # If binary classification
            if num_classes == 2:
                predictions = (predictions > 0.5).astype(int)
            else:
                predictions = np.argmax(predictions, axis=1)

            st.subheader("Predictions")
            st.write(predictions)

            # Optionally, display predictions with the original test data
            test_df['Predicted'] = predictions
            st.write(test_df)

            # Step 11: Display Confusion Matrix
            y_val_pred = model.predict(X_val)

            # If binary classification
            if num_classes == 2:
                y_val_pred = (y_val_pred > 0.5).astype(int).flatten()  # Ensure it's a 1D array
            else:
                y_val_pred = np.argmax(y_val_pred, axis=1)

            cm = confusion_matrix(y_val, y_val_pred)

            st.subheader("Confusion Matrix")
            fig_cm = plt.figure(figsize=(6, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_ if num_classes > 2 else ["0", "1"], yticklabels=le.classes_ if num_classes > 2 else ["0", "1"])
            plt.xlabel("Predicted")
            plt.ylabel("True")
            st.pyplot(fig_cm)

        # Option to download the predictions as CSV
        if uploaded_test_file is not None:
            csv = test_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")

    else:
        st.info("Please upload a CSV file to begin training.")

