from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

# Load PDF
loader = PyPDFLoader("vacation_policy.pdf")
pages = loader.load()

# Smart chunking
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
documents = splitter.split_documents(pages)

# Embedding model (384-dim)
model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

# Pinecone setup
api_key = "pcsk_P2bjo_ERwZdpSihLVaP5zv8V3YPQKzW6QdCUufvwoJ4CrkJ8vYUPaornb1uVf3GBaiYpy"
index_name = "vacation-policy"

pc = Pinecone(api_key=api_key)

# Create index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

# Upload chunks to Pinecone
for i, doc in enumerate(documents):
    vector = model.encode(doc.page_content)
    index.upsert([{
        "id": f"chunk-{i}",
        "values": vector.tolist(),
        "metadata": {
            "chunk": i,
            "text": doc.page_content,
            "source": "vacation_policy"
        }
    }])

print("✅ Upload complete: vacation policy chunks added to Pinecone.")
