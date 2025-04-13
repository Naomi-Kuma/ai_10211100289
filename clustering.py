import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

def run():
    st.title("🔄 K-Means Clustering Tool")

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
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

    if uploaded_file is not None:
        try:
            # Read the uploaded CSV file
            df = pd.read_csv(uploaded_file)

            # Check if the file is empty
            if df.empty:
                st.error("The uploaded CSV file is empty. Please upload a valid file.")
                return

            st.subheader("📋 Dataset Preview")
            st.write(df.head())

            # Step 2: Select numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

            # Check if there are numeric columns
            if len(numeric_cols) == 0:
                st.error("No numeric columns found in the dataset. Please upload a file with numeric data.")
                return

            selected_features = st.multiselect("Select 2 or 3 numeric columns for clustering", numeric_cols)

            if len(selected_features) not in [2, 3]:
                st.warning("Please select exactly 2 or 3 numeric columns.")
            else:
                X = df[selected_features]

                # Step 3: Number of clusters
                k = st.slider("Select number of clusters (k)", min_value=2, max_value=10, value=3)

                # Step 4: Run K-Means
                model = KMeans(n_clusters=k, random_state=42)
                df["Cluster"] = model.fit_predict(X)

                st.success("✅ Clustering completed!")
                st.write("Cluster centers:")
                st.write(model.cluster_centers_)

                # Step 5: Plot results with centroids
                st.subheader("🖼️ Cluster Visualization")
                fig = plt.figure()
                centers = model.cluster_centers_

                if len(selected_features) == 2:
                    sns.scatterplot(x=selected_features[0], y=selected_features[1],
                                    hue="Cluster", data=df, palette="Set2")

                    # Plot centroids
                    plt.scatter(centers[:, 0], centers[:, 1],
                                c='black', s=200, marker='X', label='Centroids')

                    plt.legend()

                else:
                    from mpl_toolkits.mplot3d import Axes3D
                    ax = fig.add_subplot(111, projection='3d')
                    ax.scatter(df[selected_features[0]], df[selected_features[1]], df[selected_features[2]],
                               c=df["Cluster"], cmap='Set2')

                    # Plot centroids
                    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
                               c='black', s=200, marker='X', label='Centroids')
                    ax.set_xlabel(selected_features[0])
                    ax.set_ylabel(selected_features[1])
                    ax.set_zlabel(selected_features[2])
                    ax.legend()

                st.pyplot(fig)

                # Optional: download clustered data
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Clustered Data", data=csv,
                                   file_name="clustered_data.csv", mime="text/csv")

        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")

    else:
        st.info("Please upload a CSV file to begin clustering.")
