# 🤖 HR Policy Chatbot

An intelligent RAG-based chatbot that answers questions from HR Policy documents using LangChain, ChromaDB, and Groq LLM.

## 🎯 Business Problem

HR teams waste hours answering repetitive policy questions from employees. This chatbot provides instant, accurate answers directly from the HR Policy PDF — saving time and improving employee experience.

## 🏗️ Architecture
PDF Document

↓

PDF Loader (PyPDF)

↓

Text Chunker (RecursiveCharacterTextSplitter)

↓

Embeddings (HuggingFace all-MiniLM-L6-v2)

↓

Vector Store (ChromaDB)

↓

Retriever (Similarity Search - Top 3 Chunks)

↓

LLM (Groq - Llama 3.3 70B)

↓

Answer
## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Framework | LangChain |
| LLM | Groq API (Llama 3.3 70B) |
| Embeddings | HuggingFace all-MiniLM-L6-v2 |
| Vector Store | ChromaDB |
| PDF Loader | PyPDFLoader |
| UI | Streamlit |
| Language | Python 3.12 |

## ✨ Features

- 📄 Loads and processes any HR Policy PDF
- 🔍 Semantic similarity search for relevant chunks
- 🤖 Accurate answers using Llama 3.3 70B
- 💬 Chat history maintained in session
- ⚡ Fast responses via Groq API

## 🚀 Setup & Run

### 1. Clone the repo
```bash
git clone https://github.com/nishant-tyagi-ai/hr-policy-chatbot-rag.git
cd hr-policy-chatbot-rag
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your API key
Create a `.env` file:GROQ_API_KEY=your_groq_api_key_here
### 4. Add your PDF
Place your HR Policy PDF as `hr_policy.pdf` in the root folder.

### 5. Run the app
```bash
python -m streamlit run app.py
```

## 📸 Sample Questions

- *"How many casual leaves are allowed per year?"*
- *"What are the working hours?"*
- *"What is the leave policy for staff?"*
- This project is for educational and demonstration purposes only. The HR policy document used belongs to its respective owner and is used only as a sample dataset

## 🙋 Author

**Nishant Tyagi**  
