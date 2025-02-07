import streamlit as st
from langchain.chains.conversation.base import ConversationChain
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from pdf_processor import PDFProcessor
from image_processor import ImageProcessor
from qa_models.openai_model import OpenAIModel
from qa_models.transformer_model import TransformerModel
from doc_processor import DocProcessor
import time
import os

# Initialize processors
pdf_processor = PDFProcessor()
image_processor = ImageProcessor()
doc_processor = DocProcessor()

# Initialize models
openai_model = OpenAIModel(api_key="YOUR_OPENAI_API_KEY")  # OpenAI GPT-3.5 Turbo
bert_model = TransformerModel()  # Transformer-based BERT model

# Initialize LangChain Memory (Persists across Streamlit reruns)
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory()

# Initialize LangChain ConversationChain for OpenAI model
api_key = os.getenv('OPENAI_API_KEY')
conversation_chain = ConversationChain(
    llm=ChatOpenAI(model="gpt-3.5-turbo", openai_api_key= api_key, temperature=0.2),
    memory=st.session_state.memory
)

st.title("📄 pdfwhiz: Conversational Q&A for PDFs, Images, and DOCX")

# File upload (PDF, Image, or DOCX)
uploaded_file = st.file_uploader("📂 Upload a PDF, Image, or DOCX file", type=["pdf", "png", "jpg", "jpeg", "docx"])

if uploaded_file:
    file_type = uploaded_file.type
    extracted_text = ""

    with st.spinner("🔍 Extracting text..."):
        time.sleep(1)  

        if "pdf" in file_type:
            extracted_text = pdf_processor.extract_text(uploaded_file)
        elif "image" in file_type:
            extracted_text = image_processor.extract_text(uploaded_file)
        elif "word" in file_type or "docx" in file_type:
            extracted_text = doc_processor.extract_text(uploaded_file)

    if extracted_text:
        question = st.text_input("💬 Ask a question about the document:")

        if question:
            with st.spinner("🤖 Generating responses..."):
                time.sleep(1)  

                # Generate responses from both models
                openai_response = conversation_chain.run(input=f"Context: {extracted_text}\n\nQuestion: {question}")
                bert_response = bert_model.get_answer(question, extracted_text)

            # Use full width instead of side-by-side layout
            st.markdown("---")  # Adds a separator line
            st.write("### 🤖 AI Model Responses")

            # OpenAI GPT Response
            st.markdown(
                f"""
                <div style="background-color: #F0F8FF; padding: 15px; border-radius: 8px; margin-bottom: 10px;">
                    <h4 style="color: #007BFF;">🟢 OpenAI GPT-3.5 Turbo</h4>
                    <p style="color: black; font-size: 16px;">{openai_response}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Transformer/BERT Response
            st.markdown(
                f"""
                <div style="background-color: #FFF5E1; padding: 15px; border-radius: 8px; margin-bottom: 10px;">
                    <h4 style="color: #FFA500;">🔵 Transformer BERT Model</h4>
                    <p style="color: black; font-size: 16px;">{bert_response}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    else:
        st.error("⚠️ Unable to extract text. Please upload a valid file.")
