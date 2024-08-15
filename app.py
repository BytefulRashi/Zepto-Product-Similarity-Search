import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, DebertaModel
import faiss
import numpy as np

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
model = DebertaModel.from_pretrained("microsoft/deberta-base")

# Function to get vector embedding
def get_vector_embedding(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    cls_embedding = last_hidden_states[:, 0, :]
    return cls_embedding.cpu().numpy()

# Load the database
database = pd.read_csv("C:/Users/rashi/Downloads/Zepto/data/flipkart_com-ecommerce_sample.csv")
database = database[:20000]

# Ensure 'merged_product_info_vector' is a list of numpy arrays
database['merged_product_info_vector'] = torch.load('C:/Users/rashi/Downloads/Zepto/vector-data-flipkart_20k.pt')
if not isinstance(database['merged_product_info_vector'].iloc[0], np.ndarray):
    database['merged_product_info_vector'] = database['merged_product_info_vector'].apply(lambda x: np.array(x))

# Convert list of tensors to a numpy array suitable for FAISS
index_vectors = np.vstack([vec for vec in database['merged_product_info_vector']]).astype('float32')

# Create FAISS index
index = faiss.IndexFlatL2(index_vectors.shape[1])
index.add(index_vectors)

# Streamlit app
st.markdown("""
    <style>
        body {
            font-family: 'Arial', sans-serif;
            background-color: #f4f4f4;
            color: #333;
        }
        .stButton > button {
            background-color: #007bff;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            border: none;
            font-size: 16px;
            cursor: pointer;
        }
        .stButton > button:hover {
            background-color: #0056b3;
        }
        .result-card {
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0px 2px 10px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        .result-card h2 {
            color: #007bff;
            font-size: 24px;
            margin-bottom: 10px;
        }
        .result-card p {
            font-size: 16px;
            line-height: 1.6;
        }
        .similarity-score {
            font-weight: bold;
            color: green;
            font-size: 18px;
        }
    </style>
""", unsafe_allow_html=True)

st.title('🛒 Product Search Engine')

query = st.text_input("🔍 Enter search query:")
if query:
    vector_query = get_vector_embedding(query)

    # Search for the top 5 similar items using FAISS
    D, I = index.search(vector_query, 5)
    similar_items = database.iloc[I[0]]

    st.markdown("## 🎯 Top Similar Products:")

    for idx, row in similar_items.iterrows():
        similarity_score = D[0][np.where(I[0] == idx)[0][0]]  # Ensure proper indexing
        st.markdown(f"""
        <div class="result-card">
            <h2>{row['product_name']}</h2>
            <p><strong>Category:</strong> {row['product_category_tree']}</p>
            <p><strong>Brand:</strong> {row['brand']}</p>
            <p><strong>Description:</strong> {row['description']}</p>
            <p><strong>Price:</strong> {row.get('price', 'N/A')}</p>
            <p class="similarity-score">Similarity Score: {similarity_score:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
