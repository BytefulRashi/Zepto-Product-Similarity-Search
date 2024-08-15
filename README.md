# Zepto Product Search Engine 🛒🔍

### Overview
This project implements a product search engine for Zepto using Natural Language Processing (NLP) techniques and a vector similarity search mechanism. The search engine enables users to input a query and retrieve the most similar products based on descriptions, categories, and other product features. The solution leverages DeBERTa embeddings and FAISS for fast and accurate similarity search.

### Features
- **Product Search**: Find the most similar products based on the query.
- **NLP-Driven**: Uses DeBERTa embeddings for precise semantic understanding.
- **Real-Time Results**: Instant retrieval of the top 5 most similar products.
- **Streamlit UI**: User-friendly interface to easily interact with the search engine.
- **Efficient Search**: FAISS indexing allows for fast similarity search, even on large datasets.

### Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Project Structure](#project-structure)
4. [Screenshots](#screenshots)
5. [Technologies Used](#technologies-used)
6. [Future Work](#future-work)
7. [Contributing](#contributing)
8. [License](#license)

---

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/Zepto-Product-Search-Engine.git
   cd Zepto-Product-Search-Engine
   ```

2. **Set up the virtual environment and install dependencies**:
   ```bash
   python3 -m venv env
   source env/bin/activate  # For Windows: env\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Download the model and vector embeddings**:
   - Install the DeBERTa model from Hugging Face:  
   ```bash
   pip install transformers
   ```
   - **(Optional)** Generate the vector embeddings using the code provided in `embedding_generator.py`.

4. **Run the Streamlit Application**:
   ```bash
   streamlit run app.py
   ```

---

### Usage

1. **Run the Streamlit Application**:
   Once you start the Streamlit app, enter a search query in the input box (e.g., "AW Bellies Footwear, Womens Footwear") and hit `Enter`.
   
2. **Retrieve Similar Products**:
   The app will display the top 5 most similar products with their details, including:
   - Product Name
   - Category
   - Brand
   - Description
   - Price
   - Similarity Score

---

### Project Structure

```bash
.
├── app.py                         # Main Streamlit application
├── embedding_generator.py         # Script to generate vector embeddings using DeBERTa
├── faiss_search.py                # FAISS similarity search code
├── requirements.txt               # Dependencies
├── data/                          # Folder for dataset (20,000 rows sample)
├── screenshots/                   # Folder for screenshots of the app
└── README.md                      # Project documentation
```

---

### Screenshots

**1. Application Interface**

![Streamlit App Interface](screenshots/app_interface.png)

**2. Query Results**

![Search Results](screenshots/search_results.png)

---

### Technologies Used

- **Python**: Programming language used to build the project.
- **Transformers (DeBERTa)**: NLP model for generating text embeddings.
- **FAISS**: Fast similarity search library for large-scale vector search.
- **Streamlit**: Framework for building the web-based user interface.
- **Pandas & NumPy**: For data manipulation and vector operations.

---

### Future Work

- **Enhanced Recommendations**: Integrate ratings, reviews, and image-based searches.
- **Performance Improvements**: Optimize FAISS for handling larger datasets efficiently.
- **Multilingual Queries**: Extend the search engine to handle queries in multiple languages.

---

### Contributing

We welcome contributions to the Zepto Product Search Engine project! Feel free to submit issues, feature requests, or pull requests.

---

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
