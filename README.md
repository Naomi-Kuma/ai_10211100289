# 🎓 AI Scholar Hub

An interactive web-based AI playground built with Streamlit to showcase hands-on applications of Machine Learning and Natural Language Processing models. AI Scholar Hub simplifies complex AI techniques into user-friendly interfaces.

---

## 📌 Features

- ✅ **Regression Analysis**: Upload datasets, specify features and targets, run Linear Regression, view performance metrics and regression line.
- 🔄 **K-Means Clustering**: Cluster data interactively, visualize in 2D or 3D, and download clustered results.
- 🧠 **Neural Networks for Classification**: Train neural networks, track accuracy/loss graphs, and make predictions.
- 💬 **LLM Q&A (RAG Approach)**: Ask natural language questions based on custom PDF documents, powered by Mistral-7B-Instruct-v0.1 via Hugging Face API.

---

## 📝 Project Motivation

The purpose of this project is to provide AI learners and students with an accessible, easy-to-use tool that enables them to explore and experiment with various AI techniques without writing complex code. It simplifies AI education through visual feedback and interactive features.

---

## 📚 Technologies Used

- **Python 3.x**
- **Streamlit** (Web Interface)
- **Scikit-Learn** (Regression, Clustering, Preprocessing)
- **TensorFlow / Keras** (Neural Networks)
- **FAISS** (Semantic Search for RAG)
- **Hugging Face Inference API** (LLM Q&A)
- **Matplotlib / Seaborn** (Visualizations)
- **Pandas / NumPy**
- **PyPDF** (Document Reading)

---

## 📊 LLM RAG Architecture

- **Vectorizer (TF-IDF)** → **FAISS Index** → Retrieve relevant document context.
- **Mistral-7B-Instruct-v0.1** via Hugging Face API → Generates answers based on retrieved context.
- Display **Response** + **Confidence Score (similarity-based)**.

---



##AUTHOR
- ** NAOMI EDEM kUMAH**
- **10211100289**
