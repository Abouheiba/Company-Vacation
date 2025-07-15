import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from pinecone import Pinecone

# --- App Setup ---
st.set_page_config(page_title="Vacation Policy Q&A", layout="centered")
st.title("📄 Company Vacation Policy Q&A")

# Load embedding model (384-dim)
@st.cache_resource
def load_embedder():
    return SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

# Load generative QA model
@st.cache_resource
def load_qa_model():
    return pipeline("text2text-generation", model="google/flan-t5-base")

embedder = load_embedder()
qa_pipeline = load_qa_model()

# Connect to Pinecone
pc = Pinecone(api_key="pcsk_P2bjo_ERwZdpSihLVaP5zv8V3YPQKzW6QdCUufvwoJ4CrkJ8vYUPaornb1uVf3GBaiYpy")
index = pc.Index("vacation-policy")

# --- UI Input ---
query = st.text_input("🔎 Ask your question (type 'end' to stop):")

# --- Handle Query ---
if query and query.lower() != "end":
    with st.spinner("Searching and answering..."):
        try:
            # Embed the question
            query_vector = embedder.encode(query).tolist()

            # Query Pinecone
            results = index.query(vector=query_vector, top_k=5, include_metadata=True)

            # Extract chunks from metadata
            chunks = [m["metadata"]["text"] for m in results["matches"]]
            context = " ".join(chunks)

            # Format input for generative model
            prompt = f"Answer the question based on the context below.\n\nContext: {context}\n\nQuestion: {query}"

            # Get the generated answer
            generated = qa_pipeline(prompt, max_length=256, do_sample=False)[0]["generated_text"]

            # Display result
            st.subheader("🧠 Answer")
            st.success(generated.strip())

            # Show source chunks
            with st.expander("📚 Source Chunks Used"):
                for i, chunk in enumerate(chunks, 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.code(chunk.strip()[:600] + "...")

            # Optional: Full context sent to model
            with st.expander("📄 Full Context Sent to Model"):
                st.code(context[:2000])

        except Exception as e:
            st.error(f"❌ Error: {e}")
