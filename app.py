import streamlit as st
from pinecone import Pinecone
from openai import OpenAI
from typing import List
from deep_translator import GoogleTranslator
import time
import math

# Debug mode
DEBUG = st.sidebar.checkbox("Debug Mode", False)

# System prompt definition remains the same
system_prompt = """You are an authoritative expert on the Gujrat Tax Law and the Ahmedabad Municipal Corporation.
[Previous system prompt content remains the same...]"""

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Initialize Pinecone with gujtaxlaw index
pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
index = pc.Index("gujtaxlaw")  # Changed to gujtaxlaw index

def get_embedding(text: str) -> List[float]:
    """Get embedding for the input text using OpenAI's embedding model."""
    # First translate Gujarati text to English if needed
    if any(ord(c) >= 0x0A80 and ord(c) <= 0x0AFF for c in text):
        try:
            text = translate_text(text, 'en')
        except Exception as e:
            if DEBUG:
                st.error(f"Translation error in get_embedding: {str(e)}")
            pass

    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

# Rest of the functions remain the same
[Previous functions: search_pinecone, translate_text, generate_response]

# Streamlit UI
st.title("ગુજરાત કર કાયદો સહાયક | Gujarat Tax Law Assistant")
st.write("કર કાયદા વિશે કોઈપણ પ્રશ્ન પૂછો | Ask any question about the Tax Law")

# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input and processing
if prompt := st.chat_input("તમે શું જાણવા માંગો છો? | What would you like to know?"):
    # Rest of the code remains the same
    [Previous chat processing logic]

# Updated sidebar with bilingual information
with st.sidebar:
    st.header("વિશે | About")
    st.write("""
    આ ચેટબોટ ગુજરાત કર કાયદા અને અમદાવાદ મ્યુનિસિપલ કોર્પોરેશન વિશે માહિતી પ્રદાન કરે છે.

    This chatbot provides information about the Gujarat Tax Law and Ahmedabad Municipal Corporation.
    """)
    st.write("""
    ભાષા સુવિધાઓ | Language Features:
    - તમે ગુજરાતી અથવા અંગ્રેજીમાં પ્રશ્નો પૂછી શકો છો
    - જો તમે ગુજરાતીમાં પૂછશો, તો તે આપોઆપ અંગ્રેજીમાં અનુવાદ થશે
    - ગુજરાતીમાં જવાબો જોવા માટે 'ગુજરાતી માં વાંચો 🔄' બટન પર ક્લિક કરો

    - You can ask questions in English or Gujarati
    - If you ask in Gujarati, it will be automatically translated to English
    - Click the 'ગુજરાતી માં વાંચો 🔄' button to see responses in Gujarati
    """)
