import streamlit as st
from pinecone import Pinecone
from openai import OpenAI
from typing import List
from deep_translator import GoogleTranslator
import time
import math

# Debug mode
DEBUG = st.sidebar.checkbox("Debug Mode", False)

# System prompt - shortened but still effective
system_prompt = """You are an expert on Gujarat Tax Law and AMC regulations. Provide:
1. Clear, concise answers with relevant citations
2. Step-by-step procedures when needed
3. References in [Source: Document, Page X] format
4. Important deadlines and requirements
Structure responses with headers and bullet points as needed."""

# Initialize clients
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
index = pc.Index("gujtaxlaw")

def chunk_text(text: str, max_tokens: int = 2000) -> List[str]:
    """Split text into smaller chunks to prevent token limit issues."""
    chars_per_chunk = max_tokens * 4  # Approximate chars per token
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        word_length = len(word) + 1  # +1 for space
        if current_length + word_length > chars_per_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = word_length
        else:
            current_chunk.append(word)
            current_length += word_length
            
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def get_embedding(text: str, max_retries: int = 3) -> List[float]:
    """Get embedding with retry logic."""
    for attempt in range(max_retries):
        try:
            if any(ord(c) >= 0x0A80 and ord(c) <= 0x0AFF for c in text):
                text = translate_text(text, 'en')
            response = client.embeddings.create(
                input=text,
                model="text-embedding-ada-002"
            )
            return response.data[0].embedding
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(1)

def search_pinecone(query: str, k: int = 3):
    """Search Pinecone with error handling."""
    try:
        query_embedding = get_embedding(query)
        results = index.query(
            vector=query_embedding,
            top_k=k,
            include_metadata=True
        )
        return results
    except Exception as e:
        if DEBUG:
            st.error(f"Pinecone search error: {str(e)}")
        return None

def generate_response(query: str, context: str, system_prompt: str):
    """Generate response with improved context handling."""
    try:
        # Limit context size
        context_chunks = chunk_text(context, max_tokens=1500)
        relevant_context = context_chunks[0] if context_chunks else ""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context: {relevant_context}\n\nQuestion: {query}"}
        ]

        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        
        response_text = response.choices[0].message.content
        
        # Translate to Gujarati
        return translate_text(response_text, 'gu')
    except Exception as e:
        if DEBUG:
            st.error(f"Response generation error: {str(e)}")
        return "માફ કરશો, જવાબ તૈયાર કરવામાં મુશ્કેલી આવી છે. કૃપા કરીને ફરી પ્રયાસ કરો."

def translate_text(text: str, target_lang: str, chunk_size: int = 3000) -> str:
    """Improved translation with chunking."""
    try:
        translator = GoogleTranslator(source='auto', target=target_lang)
        
        if len(text) <= chunk_size:
            return translator.translate(text)
            
        # Split long text into chunks
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        translated_chunks = []
        
        for chunk in chunks:
            translated_chunk = translator.translate(chunk)
            translated_chunks.append(translated_chunk)
            time.sleep(0.5)  # Rate limiting
            
        return ' '.join(translated_chunks)
    except Exception as e:
        if DEBUG:
            st.error(f"Translation error: {str(e)}")
        return text  # Return original text if translation fails

# Streamlit UI
st.title("ગુજરાત કર કાયદો સહાયક")
st.write("કર કાયદા વિશે કોઈપણ પ્રશ્ન પૂછો")

# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input and processing
if prompt := st.chat_input("તમે શું જાણવા માંગો છો?"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner('પ્રક્રિયા કરી રહ્યા છીએ...'):
        # Translate to English if needed
        search_prompt = translate_text(prompt, 'en') if any(ord(c) >= 0x0A80 and ord(c) <= 0x0AFF for c in prompt) else prompt
        
        # Search and generate response
        search_results = search_pinecone(search_prompt)
        if search_results:
            context = "\n".join([result.metadata.get('text', '') for result in search_results.matches])
            response = generate_response(search_prompt, context, system_prompt)
        else:
            response = "માફ કરશો, હાલમાં માહિતી મેળવવામાં મુશ્કેલી આવી રહી છે. કૃપા કરીને થોડી વાર પછી ફરી પ્રયાસ કરો."

        with st.chat_message("assistant"):
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar
with st.sidebar:
    st.header("વિશે")
    st.write("આ ચેટબોટ ગુજરાત કર કાયદા અને અમદાવાદ મ્યુનિસિપલ કોર્પોરેશન વિશે માહિતી પ્રદાન કરે છે.")
    st.write("ભાષા સુવિધાઓ:\n- ગુજરાતી અથવા અંગ્રેજીમાં પ્રશ્નો\n- ગુજરાતીમાં જવાબો")
