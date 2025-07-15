# 📄 Company Vacation Policy QA App

This is a **Streamlit-based question-answering app** that lets users ask questions about a company’s vacation policy using PDF documents.

It uses:
- 🤗 Hugging Face (`flan-t5-base`) for human-like answers
- 🧠 SentenceTransformers for semantic search
- 🌲 Pinecone for vector similarity search
- 🧪 Retrieval-Augmented Generation (RAG) pipeline

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/vacation-policy-qa.git
cd vacation-policy-qa
```

### 2. Set Up Virtual Environment

```bash
python -m venv venv
.env\Scriptsctivate  # On Windows
# or
source venv/bin/activate  # On Mac/Linux
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Your Pinecone API Key

Create a `.env` file with:

```ini
PINECONE_API_KEY=your_key_here
PINECONE_INDEX_NAME=vacation-policy
```

### 5. Upload PDF to Pinecone (one-time)

```bash
python main.py
```

### 6. Run the Streamlit App

```bash
streamlit run app.py
```

---

## 📦 Requirements

- Python 3.8+
- Pinecone account (free or paid)
- Hugging Face model downloads (auto)

---

## 📁 Project Structure

```
vacation-policy-qa/
├── app.py              # Streamlit frontend
├── main.py             # PDF chunking + upload
├── .env                # Pinecone keys
├── requirements.txt
└── README.md
```

---

## 🙋 FAQ

**Q:** Why are answers too short or vague?  
**A:** Try using a better embedding model (e.g. `all-MiniLM-L12-v2`, 768 dims) and `flan-t5-large`.

---

MIT License
