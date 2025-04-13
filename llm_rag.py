import streamlit as st
import requests
from pypdf import PdfReader
import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# Function to extract and clean text from PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        document_text = ""
        for page in pdf_reader.pages:
            document_text += page.extract_text()
    
    document_text = re.sub(r'\d+', '', document_text)
    document_text = re.sub(r'\s+', ' ', document_text)
    document_text = document_text.strip()
    
    return document_text

# Function to create FAISS index for document
def create_faiss_index(text):
    sentences = text.split(".")
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(sentences).toarray()

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(np.array(vectors, dtype=np.float32))
    
    return index, sentences, vectorizer

# Function to query the document using FAISS index and get similarity scores
def query_document(query, index, vectorizer, sentences):
    query_vector = vectorizer.transform([query]).toarray()
    distances, indices = index.search(np.array(query_vector, dtype=np.float32), k=3)
    
    relevant_sentences = [sentences[i] for i in indices[0]]
    scores = [1 / (1 + d) for d in distances[0]]  # Similarity score
    
    results = list(zip(relevant_sentences, scores))
    return results

# Function to query Mistral LLM via HuggingFace API
def query_mistral(prompt, context):
    api_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
    headers = {"Authorization": f"Bearer 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"}

    payload = {
        "inputs": prompt + " Context: " + context,
        "parameters": {
            "temperature": 0.5,
            "max_new_tokens": 500
        }
    }

    response = requests.post(api_url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()[0]['generated_text']
    else:
        return f"Error: {response.status_code}, {response.text}"

# Run the LLM RAG Page
def run():
    st.title("💬 LLM RAG - Q&A")

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

    st.subheader("Ask a questions about Academic City School policy and get an AI response")

    policy_pdf_path = "C:/Users/Naomi Kuma/OneDrive/Desktop/ai_project/student_policies.pdf"
                       
    document_text = extract_text_from_pdf(policy_pdf_path)
    index, sentences, vectorizer = create_faiss_index(document_text)

    if 'responses' not in st.session_state:
        st.session_state.responses = []

    for q_a in st.session_state.responses:
        st.markdown(f"**Question:** {q_a['question']}")
        st.markdown(f"**Answer:** {q_a['answer']}")
        st.markdown(f"**Confidence Score:** {q_a['top_score']:.2f}")

    unique_key = f"user_query_{len(st.session_state.responses)}"
    user_query = st.text_input("Enter your question here:", key=unique_key, value="")

    ask_button_key = f"ask_button_{len(st.session_state.responses)}"
    ask_button = st.button("Ask", key=ask_button_key)

    if ask_button and user_query.strip():
        with st.spinner("Generating response..."):
            results = query_document(user_query, index, vectorizer, sentences)
            context = " ".join([res[0] for res in results])
            response = query_mistral(user_query, context)

            st.session_state.responses.append({
                "question": user_query,
                "answer": response,
                "top_score": results[0][1]
            })

            st.markdown(f"**Answer:** {response}")
            st.markdown(f"**Confidence Score:** {results[0][1]:.2f}")

        st.text_input("Enter your question here:", key=f"user_query_{len(st.session_state.responses)}", value="")
        st.button("Ask", key=f"ask_button_{len(st.session_state.responses)}")

    st.caption("Powered by Mistral-7B-Instruct-v0.1 via HuggingFace API.")

if __name__ == "__main__":
    run()
