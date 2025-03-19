import streamlit as st
import fitz  # PyMuPDF
import pdf2image
import pytesseract
import os
import time
import requests
import uuid
import numpy as np
import io
import cv2
from PIL import Image
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from dotenv import load_dotenv
load_dotenv()
# ========== Groq API Details ==========
GROQ_API_URL = os.getenv("API_URL")
GROQ_API_KEY = os.getenv("API_KEY")

# ========== Sentence Transformer Model ==========
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(model_name)
VECTOR_SIZE = 384  # Embedding size for all-MiniLM-L6-v2

# ========== Qdrant Config ==========
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = os.getenv("QDRANT_PORT")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

# Set the path to Tesseract executable - UPDATE THIS PATH AS NEEDED
Tessaract_path = os.getenv("TESSERACT_PATH")
pytesseract.pytesseract.tesseract_cmd = r'Tessaract_path'

# Initialize Qdrant client
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# Create collection if it doesn't exist
try:
    qdrant_client.get_collection(COLLECTION_NAME)
except Exception:
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
    )

# ========== Streamlit Page Config ==========
st.set_page_config(page_title="PDF to Qdrant + Groq RAG System", page_icon="📄")

# ========== State Management ==========
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'upload'
if 'use_ocr' not in st.session_state:
    st.session_state.use_ocr = False

# ========== Function: Preprocess Image for OCR ==========
def preprocess_image(image):
    # Convert PIL Image to OpenCV format
    img = np.array(image)
    
    # Convert to grayscale if it's not already
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Apply binary thresholding
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Convert back to PIL Image
    return Image.fromarray(img)

# ========== Function: Extract Text with OCR Support ==========
def extract_text_from_pdf(pdf_file, use_ocr=False, dpi=300):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    
    # Try regular text extraction first
    for page in doc:
        page_text = page.get_text()
        if page_text.strip():  # If there's actual text content
            text += page_text
    
    # If no text was found or OCR is forced, try OCR
    if (not text.strip() or use_ocr) and use_ocr:
        text = ""  # Reset text if we're using OCR
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Render at high resolution
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
            
            # Convert pixmap to PIL Image
            img = Image.open(io.BytesIO(pix.tobytes()))
            
            # Preprocess the image
            preprocessed_img = preprocess_image(img)
            
            # Apply OCR
            config = f'--oem 3 --psm 3 -l eng'
            page_text = pytesseract.image_to_string(preprocessed_img, config=config)
            text += page_text + "\n\n"
    
    return text

# ========== Function: Generate Embeddings ==========
def generate_embedding(text):
    # Using sentence-transformers to generate embeddings
    embedding = embedding_model.encode(text)
    return embedding.tolist()

# ========== Function: Store in Qdrant ==========
def store_in_qdrant(text, file_name):
    # Split text into chunks (simplified)
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
    
    points = []
    for i, chunk in enumerate(chunks):
        chunk_id = str(uuid.uuid4())
        embedding = generate_embedding(chunk)
        
        point = PointStruct(
            id=chunk_id,
            vector=embedding,
            payload={
                "text": chunk,
                "file_name": file_name,
                "chunk_index": i
            }
        )
        points.append(point)
    
    # Store in Qdrant
    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    
    return len(points)

# ========== Function: Search in Qdrant ==========
def search_in_qdrant(query, limit=5):
    query_embedding = generate_embedding(query)
    
    search_results = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=limit
    )
    
    context = ""
    for result in search_results:
        context += result.payload["text"] + "\n\n"
    
    return context

# ========== Function: Call Groq API ==========
def call_groq_api(user_question, context):
    headers = {
        'Authorization': f'Bearer {GROQ_API_KEY}',
        'Content-Type': 'application/json'
    }
    payload = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "system", "content": "You are a helpful educational assistant. Answer questions directly and concisely based ONLY on the provided context. Do not add external knowledge. Keep responses brief but complete. Never show your reasoning process or use phrases like 'based on the context' or 'the context mentions.' Simply provide the answer as if you naturally know it. Do not use <think> tags or similar markers. And if a user asks you to explain the topic in detail then use the information or chunkd you get and then explain it in very easy way and in detail to the user i.e. explain this to me in detail or make me understand it better or explain it to me in about x words then you have to explain that thing to the user in very simpler way.Also if the question is if a girl's height is of 36.98 m . and her friend's height is of 69.40m .convert into the smallest unit of measurement, then first find the smallest unit of measurment and then calculate the height with formula."},
            {"role": "user", "content": f"Question: {user_question}\n\nContext: {context}"}
        ],
        "stream": False  # Changed to non-streaming for simplicity
    }

    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            st.error(f"Error from Groq API: {response.status_code} - {response.text}")
            return f"Error: Could not get a response from the API. Status code: {response.status_code}"
    except Exception as e:
        st.error(f"Exception occurred: {str(e)}")
        return f"Error: An exception occurred while processing your question: {str(e)}"

# ========== Page 1: Upload PDF ==========
if st.session_state.current_page == 'upload':
    st.title("🧠 NeuroScholar 🎓")
    st.subheader("Upload PDF Files")

    # OCR option
    st.session_state.use_ocr = st.checkbox("Use OCR for text extraction (for scanned PDFs)", value=st.session_state.use_ocr)
    
    if st.session_state.use_ocr:
        ocr_dpi = st.slider("OCR Resolution (DPI)", min_value=150, max_value=600, value=300, step=150)
    else:
        ocr_dpi = 300  # Default value
    
    uploaded_files = st.file_uploader("Upload your PDF files", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing {uploaded_file.name}...")
            
            try:
                # Extract text (with OCR if enabled)
                text = extract_text_from_pdf(uploaded_file, use_ocr=st.session_state.use_ocr, dpi=ocr_dpi)
                
                if not text.strip():
                    st.warning(f"No text could be extracted from {uploaded_file.name}. The file might be protected or contains only images.")
                    continue
                
                # Store in Qdrant
                chunks_stored = store_in_qdrant(text, uploaded_file.name)
                
                # Store in session state
                st.session_state.uploaded_files.append({
                    'name': uploaded_file.name,
                    'text': text[:100] + "...",  # Only store preview
                    'chunks': chunks_stored
                })
                
                # Update progress
                progress_bar.progress((i + 1) / len(uploaded_files))
                status_text.text(f"Processed {uploaded_file.name} - Stored {chunks_stored} chunks in Qdrant")
            
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        
        progress_bar.progress(1.0)
        st.success("Files Saved Successfully")
        
        if st.button("Chat Now !!"):
            st.session_state.current_page = 'chat'
            st.rerun()

# ========== Page 2: Chat Interface ==========
elif st.session_state.current_page == 'chat':
    st.title("💬 Ask Neuro")
    st.sidebar.write("📁 Uploaded PDFs:")
    for file in st.session_state.uploaded_files:
        st.sidebar.write(f"- {file['name']} ({file.get('chunks', 'N/A')} chunks)")

    user_question = st.text_input("Ask any question from your PDFs")

    if user_question:
        # Append user question to chat
        st.session_state.chat_history.append({"role": "user", "content": user_question})

        # Display the question
        with st.chat_message("user"):
            st.write(user_question)
            
        # Show spinner while getting response
        with st.spinner("Getting answer..."):
            # Retrieve relevant context from Qdrant
            context = search_in_qdrant(user_question)
            
            # Get response from Groq
            response = call_groq_api(user_question, context)
        
        # Display response
        with st.chat_message("assistant"):
            st.write(response)
        
        # Append assistant response to chat
        st.session_state.chat_history.append({"role": "assistant", "content": response})

    # Display chat history
    st.write("Chat History:")
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.write(chat["content"])

    # Clear chat history button
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

    # Download Chat History
    if st.button("Download Chat History"):
        history_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.chat_history])
        st.download_button("Download", history_text, file_name="chat_history.txt")

    # Back Button
    if st.button("Back to Upload"):
        st.session_state.current_page = 'upload'
        st.rerun()